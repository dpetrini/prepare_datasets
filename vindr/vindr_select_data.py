#
# Filter data from VINDR-MAMMO - Baseado nos processos do CBIS
#
# Separates images and corresponding masks according to a criteria HARDCODED HERE and input
# xls (mass or calcification).
#
# Read dicom, convert to PNG and crop exam image.
# Selects randomly a number of images. Save the images and masks with meaningfull dir name.
# Generate statistics for selection.
#
#  Generate masks for patch classifier (reading coordinates from csv) -> faster to compute in
#  training.
#
#  Ex.
#  -processar os benign
#  python3 vindr_select_data.py -f './csv/vindr-mammo/breast-level_annotations.csv' -a b -o './data-vindr' -n 0
#
#  -processar os malign (precisa assim os 2 tipos)
#  python3 vindr-select-data.py -f './csv/vindr-mammo/breast-level_annotations.csv' -a m -o './data-vindr' -n 0
#
#  Obs. Para massas geralmente o tamanho de algumas imagens nao bate com as mascaras. Entao 
#       selecionar n maior e depois apagar algumas. Para calcificacoes nao tem esse problema
#       usar n desejado.
#
# DGPP - 31/08/2022 - Initial version for Unet studies (patches)
#        18/09/2022 - Adicionado pre-processamento (process.ipynb) & multiProcess


import pandas as pd
import numpy as np
import argparse
import sys
import random
from multiprocessing import Pool
from functools import partial


import png
import pydicom
import os
import cv2
import time

from crop_mammogram import crop_mammogram_one_image

from convert_dicom import convert_dicom_to_png
# from prepare_data.multiprocess import NUM_PROCESS

# for multiprocess
NUM_PROCESSES = 6           # 4 funcionou bem, talvez 6 se desligar browser

DEFAULT_SIZE = (896, 1152)   # (896, 1152) always. (1792, 2304) para TPU, Sergio.

RESIZE_EXPAND_CHANNELS = True
SKIP_IMG_GENERATION = False
SKIP_MASK_GENERATION = False

N_SIGMA = 4             # numero de sigmas a manter na imagem

def save_dicom_image_as_png_vindr(dicom_filename, png_filename, bitdepth=12):
    """
    :param dicom_filename: path to input dicom file.
    Save 12-bit mammogram from dicom as rescaled 16-bit png file.
    :param png_filename: path to output png file.
    :param bitdepth: bit depth of the input image. Set it to 12 for 12-bit mammograms.
    """
    png_img = convert_dicom_to_png(dicom_filename)
    cv2.imwrite(png_filename, png_img)

def save_dicom_image_as_png(dicom_filename, png_filename, side, bitdepth=12):
    """
    :param dicom_filename: path to input dicom file.
    Save 12-bit mammogram from dicom as rescaled 16-bit png file.
    :param png_filename: path to output png file.
    :param bitdepth: bit depth of the input image. Set it to 12 for 12-bit mammograms.
    """
    data = pydicom.read_file(dicom_filename)
    image = data.pixel_array

    depth_allocated = data.data_element('BitsAllocated').value
    depth_stored = data.data_element('BitsStored').value
    photometricInterpretation = data.data_element('PhotometricInterpretation').value

    if depth_allocated != 16 or depth_stored not in [12, 16]:
        print('Invalid depth: ', depth_allocated, depth_stored)

    # Pre-processamento da imagem # (vide notebook process.ipynb) *************

    # Monochrome1: minimum=white, so need to invert
    # https://dicom.innolitics.com/ciods/rt-dose/image-pixel/00280004
    if str(photometricInterpretation) == 'MONOCHROME1':
        image = np.invert(image)

    min = image.min()
    max = image.max()

    # Remove marca L-CC, etc sempre no canto superior contrario ao lado
    LIMIT_Y = int(0.12*(image.shape[0]))
    LIMIT_X = int(0.28*(image.shape[1]))

    # Seleciona o lado
    if side == 'R':
        image[0:LIMIT_Y, 0:LIMIT_X] = min
    elif side == 'L':
        image[0:LIMIT_Y, (image.shape[1] - LIMIT_X):image.shape[1]] = min

    # Encontra limiar OTSU
    limiar, _ = cv2.threshold(image, 0, max, cv2.THRESH_OTSU)
    # troca o minimo da imagem para limiar, para melhorar a normal da imagem
    image[image < limiar] = limiar

    # incluir processamento de 4*sigmas para extrair limites extremos
    image_flat = image.flatten()
    n_sigma = N_SIGMA

    # Remove the limiar
    rm = np.array([limiar])
    # np.in1d return true if the element of `a` is in `rm`
    idx = np.in1d(image_flat, rm)
    clean_array = image_flat[~idx]

    # Obtain normal parameters and find the limits based in N_SIGMA*sigma
    mu, sigma = np.mean(clean_array), np.std(clean_array)
    limite_inf = int(mu-n_sigma*sigma)
    limite_sup = int(mu+n_sigma*sigma)

    image[image < limite_inf] = limite_inf
    image[image > limite_sup] = limite_sup

    # Fim pre-processamento     ***********************************************

    # Salva PNG conforme numero de bits lido (default compression=None)
    with open(png_filename, 'wb') as f:
        writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=int(depth_stored), greyscale=True)
        writer.write(f, image.tolist())

# Moved functions below from middle o file to here
def resize_expand_channels(img_file, expand=True):
    """ Resize to our default size and replicate channels to ease dataloaders """
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, DEFAULT_SIZE, interpolation=cv2.INTER_AREA)
    if (expand):
        image = np.zeros((*img.shape[0:2], 3), dtype=np.uint16)
        image[:, :, 0] = img
        image[:, :, 1] = img
        image[:, :, 2] = img
    else:
        image = img
    os.remove(img_file)
    cv2.imwrite(img_file, image)

# Globals & DEFINES

# indicate here where dataset is
TOP_VINDR = '/media/dpetrini/DISK041/datasets/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images'
sufix = random.randint(1,200)
TEMP_DIR = './'+str(sufix)
save_path = '/media/dpetrini/PETRINI/TPU/full'   #not reading from parameter here

file_list = []

def convert_one_image(sel, resize_expand=True):
    """ Convert image to dicom and prepare for dataloaders """

    if file_list[sel]['short_case_id'] == 'dbca9d28b' and file_list[sel]['side'] == 'L':
        print('Anomalia VINDR (2x L-CC). Skipping: ', file_list[sel])
        return

    image_source = (os.path.join(TOP_VINDR, file_list[sel]['dir'], file_list[sel]['file']))+'.dicom'
    split = 'train' if file_list[sel]['split']=='training' else 'test'
    dir_destiny = os.path.join(save_path, split, file_list[sel]['type'])
    image_name = file_list[sel]['short_case_id']+'_'+file_list[sel]['side']+'_'+ \
                 file_list[sel]['view']+'_B'+file_list[sel]['birads']+ \
                 '_D'+file_list[sel]['density']+'.png'
    destiny_path = os.path.join(dir_destiny, image_name)
    print(destiny_path)

    # se já processamos imagem sai
    if os.path.isfile(destiny_path):
        print('Done.')
        return

    # try:
    save_dicom_image_as_png(image_source, destiny_path, file_list[sel]['side'] ,16)
    if RESIZE_EXPAND_CHANNELS:
        resize_expand_channels(destiny_path, False)
    # except:
    #     print('Erro ao processar arquivo: ', image_source)


# Criamos assim como main() em 11-2022 - nao testado
def main():
    
    # MAIN
    ap = argparse.ArgumentParser(description='Convert ddsm csv in NYU label format.')
    ap.add_argument("-f", "--file",  required=True,
                    help="csv file to convert.")
    ap.add_argument("-p", "--path", required=False,
                    help="folder to store [CROPPED] png converted images")
    ap.add_argument("-o", "--outputpath", required=False,
                    help="folder to store png converted images")
    ap.add_argument("-n", "--nfiles", required=False,
                    help="number of files to generate. Zero means all.")
    ap.add_argument("-c", "--crop", dest='crop', default=False, action='store_true',
                    help="to crop mammogram include -c")
    ap.add_argument("-a", "--abnormality", required=True,
                    help="abnormality type b: benign, m: malignant")

    args = vars(ap.parse_args())

    print(args)

    if (args['file'].split('.')[-1] != 'csv'):
        print('Input must be csv file.')
        sys.exit()

    # get number of files to select n
    if args['nfiles'] is None:
        n_files = 0
        print('Number of files not defined, selecting all.')
    else:
        n_files = int(args['nfiles'])

    # if requested mark for cropping
    do_crop = args['crop']

    # manage crop folder, create if need, erase if exist files, etc
    if do_crop:
        if args['path'] is None:
            cropped_file_path = './cropped'
            print('Using default path: ./cropped')
        else:
            if os.path.isdir(args['path']) is False:
                print('Destination cropped and cropped_mask dirs does not exist - trying to create')
                os.makedirs(args['path'])
            else:
                old_files = os.listdir(args['path'])
                if len(old_files) > 0:
                    print('Erasing old contents of ', args['path'])
                    for f in old_files:
                        os.remove(os.path.join(args['path'], f))
            cropped_file_path = args['path']

        # do same for mask path
        if os.path.isdir(args['path']+'_mask') is False: 
            os.makedirs(args['path']+'_mask')   # create mask path
        else:
            for f in os.listdir(args['path']+'_mask'):  # exist then erase files
                os.remove(os.path.join(args['path']+'_mask', f))
        cropped_mask_path = args['path']+'_mask'

    # Output dir to store png full images and masks
    if args['outputpath'] is None:
        if os.path.isdir(TEMP_DIR) is False:
            os.makedirs(TEMP_DIR, exist_ok=False)
        save_path = TEMP_DIR
        save_path_mask = save_path+'_mask'
    else:
        if not os.path.isdir(args['outputpath']):
            print('Creating output path', args['path'])
            os.makedirs(args['outputpath'], exist_ok=False)
        save_path = args['outputpath']
        save_path_mask = save_path+'_mask'

        # Cria subdiretorios adicionais se necessario
        os.makedirs(os.path.join(save_path, 'train', 'malign',), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'test' , 'malign',), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'train', 'benign',), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'test' , 'benign',), exist_ok=True)


    if os.path.isdir(save_path_mask) is False:
        os.makedirs(save_path_mask, exist_ok=False)

    if args['abnormality'] is not None:
        if args['abnormality'] == 'b':
            abnormality = 'BENIGN'
            print('Selecting Benign exams from file ', args['file'])
        elif args['abnormality'] == 'm':
            abnormality = 'MALIGNANT'
            print('Selecting Malignant exams from file ', args['file'])
        else:
            print('Abnormality must be b or m')
            sys.exit()

    # read to pandas frame
    df = pd.read_csv(args['file'])

    print('Shape before:', df.shape)

    # transforma em np para manipular
    table = np.array(df)

    print('Shape depois:', table.shape)

    # file_list = []  Became global

    # Following lines filter what we want. Hardcoded now- -CRIE SUA SELECAO AQUI
    count = 0
    for i in range(table.shape[0]):
        if abnormality == 'MALIGNANT':
            if table[i][7] == 'BI-RADS 3' or table[i][7] == 'BI-RADS 4' or table[i][7] == 'BI-RADS 5':
                file_list.append({'dir': table[i][0], 'file': table[i][2],
                                'birads': str(table[i][7])[-1],
                                'density': table[i][8][-1],
                                'side': table[i][3], 'view': table[i][4],
                                'split': table[i][9],
                                'short_case_id':  table[i][0][0:9],
                                'type': 'malign'})
                count = count+1
        elif abnormality == 'BENIGN':
            if table[i][7] == 'BI-RADS 1' or table[i][7] == 'BI-RADS 2':
                file_list.append({'dir': table[i][0], 'file': table[i][2],
                                'birads': str(table[i][7])[-1],
                                'density': table[i][8][-1],
                                'side': table[i][3], 'view': table[i][4],
                                'split': table[i][9],
                                'short_case_id':  table[i][0][0:9],
                                'type': 'benign'})
                count = count+1

    print('Current selection (all): ', count, len(file_list))

    # if n_files == 0 then use all files
    if n_files == 0:
        n_files = len(file_list)

    if len(file_list) < n_files:
        print('Number of files requested:', n_files, 'bigger than available in current filter:', len(file_list))
        sys.exit()

    selection = list(range(0, n_files))
    print ("Selection of N samples: ", selection, len(selection))

    # pegar n imagens e copia para outro diretorio com nome obtido do path
    # necessario fazer essa busca porque o caminho informado no csv nao eh correto 
    print('Converting images to png: ', len(file_list))


    #erro em a235482f7_R_MLO_B2_DC.


    # Versão original de processamento sequencial
    # # def convert_many_images():
    # # entao fazer serialmente:
    # count = 0
    # count_train = 0
    # if not SKIP_IMG_GENERATION:
    #     for image in selection:
    #         if file_list[image]['short_case_id'] == 'dbca9d28b' and file_list[image]['side'] == 'L':
    #             print('Anomalia VINDR (2x L-CC). Skipping: ', file_list[image])
    #             continue
    #         convert_one_image(image, RESIZE_EXPAND_CHANNELS)
    #         count = count+1
    #         if file_list[image]['split']=='training':
    #             count_train += 1
    # else:
    #     print('Skipping image generation.')


    # Fazendo por processamento paralelo - muito mais rápido
    f = partial(convert_one_image, resize_expand=RESIZE_EXPAND_CHANNELS)
    p = Pool(NUM_PROCESSES)
    # Execute for all in the list
    p.map(f, selection)

    print('Converted(all): ', count)

    sys.exit()


# colocamos aqui - 11-2022
if __name__ == '__main__':
    main()