#
# Filter data from VINDR-MAMMO - Based in CBIS processing
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
#  -processar os benign (fazendo todos arquivos n=0)
#  python3 vindr_select_data.py -f './csv/vindr-mammo/breast-level_annotations.csv' -a b -o './data-vindr' -n 0
#
#  -processar os malign (precisa assim os 2 tipos), fazendo 10 arquivos (n=10)
#  python3 vindr-select-data.py -f './csv/vindr-mammo/breast-level_annotations.csv' -a m -o './data-vindr' -n 10
#
#  -process chest files (many corrupted in kaggle dataset download ??)
#  python3 vindr_select_data.py -f './csv/vindr-mammo/train.csv' -a m -o './data-vindr' -n 20  (only 12 ok)
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

from process_image import data_preprocess

# indicate here where dataset is
# TOP_VINDR = '/media/dpetrini/DISK041/datasets/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images'
TOP_VINDR = '/media/dpetrini/DISK041/datasets/vinbigdata-chest-xray-abnormalities-detection/train'

TYPE = 'CHEST'  # 'CHEST' 'MAMMO' 

# for multiprocess
NUM_PROCESSES = 6          # 4 funcionou bem, talvez 6 se desligar browser

DEFAULT_SIZE = (896, 1152)

RESIZE_EXPAND_CHANNELS = True


def save_dicom_image_as_png(dicom_filename, png_filename, side, bitdepth=12, debug=False):
    """
    :param dicom_filename: path to input dicom file.
    Save 12-bit mammogram from dicom as rescaled 16-bit png file.
    :param png_filename: path to output png file.
    :param bitdepth: bit depth of the input image. Set it to 12 for 12-bit mammograms.
    """
    data = pydicom.read_file(dicom_filename)

    # Pre-processamento da imagem # (vide notebook process.ipynb) *************
    image, depth_stored = data_preprocess(data, side, debug)
 
    # Salva PNG conforme numero de bits lido (default compression=None)
    with open(png_filename, 'wb') as f:
        writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=int(depth_stored), greyscale=True)
        writer.write(f, image.tolist())

    return depth_stored

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


def convert_one_image(sel, file_list=[], save_path='./', resize_expand=True):
    """ Convert image to dicom and prepare for dataloaders """

    side = None

    if TYPE == 'MAMMO':
        if file_list[sel]['short_case_id'] == 'dbca9d28b' and file_list[sel]['side'] == 'L':
            print('Anomalia VINDR (2x L-CC). Skipping: ', file_list[sel])
            return

        image_source = (os.path.join(TOP_VINDR, file_list[sel]['dir'], file_list[sel]['file']))+'.dicom'
        split = 'train' if file_list[sel]['split']=='training' else 'test'
        dir_destiny = os.path.join(save_path, split, file_list[sel]['type'])
        image_name = file_list[sel]['short_case_id']+'_'+file_list[sel]['side']+'_'+ \
                    file_list[sel]['view']+'_B'+file_list[sel]['birads']+ \
                    '_D'+file_list[sel]['density']+'.png'
        side = file_list[sel]['side']
    elif TYPE == 'CHEST':
        image_source = (os.path.join(TOP_VINDR, file_list[sel]['image_id']))+'.dicom'
        split = 'train'     # always in this case
        dir_destiny = os.path.join(save_path, split, file_list[sel]['type'])
        image_name = file_list[sel]['image_id']+'.png'

    destiny_path = os.path.join(dir_destiny, image_name)
    print(destiny_path)

    # se já processamos imagem sai
    if os.path.isfile(destiny_path):
        print('Done.')
        return

    try:
        save_dicom_image_as_png(image_source, destiny_path, side, 16)
        if resize_expand:
            resize_expand_channels(destiny_path, False)
    except:
        print('Erro ao processar arquivo: ', image_source)


# MAIN
def main():
    sufix = random.randint(1,200)
    TEMP_DIR = './'+str(sufix)

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
        elif args['abnormality'] == 'a':
            print('Selecting ALL exams from file ', args['file'])
            abnormality = 'ALL'
        else:
            print('Abnormality must be b, m or a')
            sys.exit()

    # read to pandas frame
    df = pd.read_csv(args['file'])
    # transforma em np para manipular
    table = np.array(df)

    print(len(table), table)

    file_list = []

    # Following lines filter what we want. Hardcoded now- -CRIE SUA SELECAO AQUI
    count = 0
    for i in range(table.shape[0]):
        if TYPE == 'MAMMO':
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
        elif TYPE == 'CHEST':               # all train here
            if abnormality == 'MALIGNANT':
                if table[i][1] != 'No finding':
                    file_list.append({'image_id': table[i][0],
                                    'class_name': table[i][1],
                                    'class_id': table[i][2],
                                    'rad_id': table[i][3],
                                    'xmin': float(table[i][4]),
                                    'ymin': float(table[i][5]),
                                    'xmax': float(table[i][6]),
                                    'ymax': float(table[i][7]),
                                    'type': 'malign'})
                    count = count+1
            if abnormality == 'BENIGN':
                if table[i][1] == 'No finding':
                    file_list.append({'image_id': table[i][0],
                                    'class_name': table[i][1],
                                    'class_id': table[i][2],
                                    'rad_id': table[i][3],
                                    'type': 'benign'})
                    count = count+1
            if abnormality == 'ALL':
                file_list.append({'image_id': table[i][0],
                                'class_name': table[i][1],
                                'class_id': table[i][2],
                                'rad_id': table[i][3]})
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
    print('Converting images to png: ', len(file_list), ' using: ', NUM_PROCESSES, ' parallel processes.')

    #erro em a235482f7_R_MLO_B2_DC.

    # Fazendo por processamento paralelo - muito mais rápido
    f = partial(convert_one_image, file_list=file_list, save_path=save_path, resize_expand=RESIZE_EXPAND_CHANNELS)
    p = Pool(NUM_PROCESSES)
    # Execute for all in the list
    p.map(f, selection)

    print('Converted(all): ', count)


if __name__ == '__main__':
    main()