#
# Filter data from VINDR-MAMMO - Baseado nos processos do CBIS - Criação de patches
#
# Separates images and corresponding masks according to a criteria HARDCODED HERE and input
# xls (mass or calcification).
#
# Read dicom, convert to PNG and crop exam image.
#
#  Generate masks for patch classifier (reading coordinates from csv)
#
#   Supports patches for Self Supervised Learning - no labels - CHECK CODE TO SUPPORT SSL BELOW (comments)
#
#  Ex.
#  -processar os malign (precisa assim os 2 tipos)
#   python3 vindr-select-data_patch.py -f './csv/vindr-mammo/finding_annotations.csv' -a m -o './data-vindr-patch' -n 0
#
#
# DGPP - Nov/2022 - Created file based in CBIS S10 process. Core refactor - separate patch only routines
#        Abr/2023 - Support different patch sizes


import pandas as pd
import numpy as np
import argparse
import sys
import random
from multiprocessing import Pool
from functools import partial
import os
import cv2
import time

from process_patch import extract_s10_patches, extract_bg_patches
from vindr_select_data import save_dicom_image_as_png

# for multiprocess
NUM_PROCESSES = 6           # 4 funcionou bem, talvez 6 se desligar browser
PATCH_SIZE = 224            # Default = 224, 112, 448 (TPU)
DEFAULT_SIZE = (896, 1152)  # (896, 1152) always. (1792, 2304) para TPU, Sergio.
RESIZE_EXPAND_CHANNELS = True
NON_BLACK_LIMIT = 0  # 1000 # limite superior de pixels pretos para background

# indicate here where dataset is
TOP_VINDR = '/media/dpetrini/DISK041/datasets/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images'
sufix = random.randint(1,200)
TEMP_DIR = './'+str(sufix)


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

    return image
    # cv2.imwrite(img_file, image)


def prepare_category_sufix(finding):
    sufix = '_'
    if 'Mass' in finding:
        sufix += 'M_'
    if 'Global Asymmetry' in finding:
        sufix += 'GA_'
    if 'Architectural Distortion' in finding:
        sufix += 'AD_'
    if 'Nipple Retraction' in finding:
        sufix += 'NR_'
    if 'Suspicious Calcification' in finding:
        sufix += 'SC_'
    if 'Focal Asymmetry' in finding:
        sufix += 'FA_'
    if 'Skin Thickening' in finding:
        sufix += 'ST_'
    if 'Asymmetry' in finding:
        sufix += 'AS_'
    if 'Suspicious Lymph Node' in finding:
        sufix += 'LN_'
    return sufix


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
ap.add_argument("-a", "--abnormality", required=False,
                help="abnormality type b: benign, m: malignant")

args = vars(ap.parse_args())

print(args)

if (args['file'].split('.')[-1] != 'csv'):
    print('Input must be csv file.')
    sys.exit()

# get number of files to select n
if args['nfiles'] is None:
    n_files = 0
    print('Number of files not defined, selecting all (SELF_SUP).')
else:
    n_files = int(args['nfiles'])


# Output dir to store png full images and masks
if args['outputpath'] is None:
    if os.path.isdir(TEMP_DIR) is False:
        os.makedirs(TEMP_DIR, exist_ok=False)
    save_path = TEMP_DIR
else:
    if not os.path.isdir(args['outputpath']):
        print('Creating output path', args['path'])
        os.makedirs(args['outputpath'], exist_ok=False)
    save_path = args['outputpath']

    if args['abnormality'] is not None:
        # Cria subdiretorios adicionais se necessario
        os.makedirs(os.path.join(save_path, 'train', 'malign_patch',), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'train', 'malign_bg',), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'test' , 'malign_patch',), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'test' , 'malign_bg',), exist_ok=True)


if args['abnormality'] is not None:
    if args['abnormality'] == 'b':
        abnormality = 'BENIGN'
        print('Selecting Benign exams from file ', args['file'])
    elif args['abnormality'] == 'm':
        abnormality = 'MALIGNANT'
        print('Selecting Malignant exams from file ', args['file'])
else:
    print('Abnormality must be b or m. None means using all files.')
    print('Not creating dirs, all files in input dir')
    abnormality = 'ALL'
    # save_path += '_patch_SSL'
    # if os.path.isdir(save_path) is False:
    #     os.makedirs(save_path, exist_ok=False)

# read to pandas frame
df = pd.read_csv(args['file'])

print('Shape before:', df.shape)

# transforma em np para manipular
table = np.array(df)

print('Shape depois:', table.shape)

file_list = []
mask_file_list = []

start_time = time.time()

# Following lines filter what we want. Hardcoded now- -CRIE SUA SELECAO AQUI
count = 0
for i in range(table.shape[0]):
    if abnormality == 'MALIGNANT':
        if table[i][9] != '[\'No Finding\']':
            file_list.append({'dir': table[i][0], 'file': table[i][2],
                              'birads': str(table[i][7])[-1],
                              'density': table[i][8][-1],
                              'side': table[i][3], 'view': table[i][4],
                              'split': table[i][15],
                              'short_case_id':  table[i][0][0:9],
                              'category': table[i][9],
                              'height': int(table[i][5]),
                              'width': int(table[i][6]),
                              'xmin': float(table[i][11]),
                              'ymin': float(table[i][12]),
                              'xmax': float(table[i][13]),
                              'ymax': float(table[i][14]),
                              'type': 'malign'})
            count = count+1
    if abnormality == 'ALL':
        file_list.append({'dir': table[i][0], 'file': table[i][2],
                            'birads': str(table[i][7])[-1],
                            'density': table[i][8][-1],
                            'side': table[i][3], 'view': table[i][4],
                            'split': table[i][9],
                            'short_case_id':  table[i][0][0:9],
                            'type': 'all'})
        count = count+1


# print('Current selection (all): ', count, len(file_list))

# if n_files == 0 then use all files
if n_files == 0:
    n_files = len(file_list)

if len(file_list) < n_files:
    print('Number of files requested:', n_files, 'bigger than available in current filter:', len(file_list))
    sys.exit()

selection = list(range(0, n_files))
# print ("Selection of N samples: ", selection, len(selection))


# pegar n imagens e copia para outro diretorio com nome obtido do path
# necessario fazer essa busca porque o caminho informado no csv nao eh correto 
print(f'\nCreating patches ({PATCH_SIZE}x{PATCH_SIZE}) by reading: {n_files} full imgs out of {len(file_list)}\n')


debug = False

# Function below for S10 Sinai-paper like patches
def convert_one_image(sel, resize_expand=True):
    """ Convert image to dicom and extract patches """

    image_source = (os.path.join(TOP_VINDR, file_list[sel]['dir'], file_list[sel]['file']))+'.dicom'
    split = 'train' if file_list[sel]['split']=='training' else 'test'
    dir_destiny = os.path.join(save_path, split, file_list[sel]['type'])

    image_name = file_list[sel]['short_case_id']+'_'+file_list[sel]['side']+'_'+ \
                 file_list[sel]['view']+'_B'+file_list[sel]['birads'] + \
                 '_D'+file_list[sel]['density']+'.png'

    temp_image_path = os.path.join('./temp', image_name)
    dir_destiny_patch = dir_destiny + '_patch'
    dir_destiny_mask = dir_destiny + '_mask'    # nao usa aqui
    dir_destiny_bg = dir_destiny + '_bg'

    sufix = prepare_category_sufix(file_list[sel]['category'])

    print(temp_image_path, split, sufix)

    # We must make same process as image: save converted dicom then patch from it.
    save_dicom_image_as_png(image_source, temp_image_path, file_list[sel]['side'] ,16)
    if RESIZE_EXPAND_CHANNELS:
        image = resize_expand_channels(temp_image_path, False)

    # Agora aqui vamos fazer lógica do tamanho da imagem, patch e salvar o patch
    height = DEFAULT_SIZE[1]
    width = DEFAULT_SIZE[0]
    h_rel = height/file_list[sel]['height']
    w_rel = width/file_list[sel]['width']

    xmin_f = int(file_list[sel]['xmin'] * w_rel)
    xmax_f = int(file_list[sel]['xmax'] * w_rel)
    ymin_f = int(file_list[sel]['ymin'] * h_rel)
    ymax_f = int(file_list[sel]['ymax'] * h_rel)

    h = ymax_f-ymin_f
    w = xmax_f-xmin_f

    if debug:
        print(h, w, file_list[sel]['category'])
        patch = image[ymin_f:ymax_f, xmin_f:xmax_f]
        cv2.imshow('Real Patch', patch)
        cv2.waitKey(0)

    patch_list, mask_list, mask_bg = extract_s10_patches(image, None, xmin_f, ymin_f, w, h, ps=PATCH_SIZE, debug=debug)

    for i, p in enumerate(patch_list):
        cv2.imwrite(os.path.join(dir_destiny_patch, image_name.split('.')[0]+'_patch_s10'+sufix+str(i)+'.png'), p)

    if len(mask_list):
        for i, m in enumerate(mask_list):
            cv2.imwrite(os.path.join(dir_destiny_mask, image_name.split('.')[0]+'_mask_patch_s10_'+str(i)+'.png'), m)

    bg_list = extract_bg_patches(image, mask_bg, ps=PATCH_SIZE, limit=NON_BLACK_LIMIT)

    for i, bg in enumerate(bg_list):
        cv2.imwrite(os.path.join(dir_destiny_bg, image_name.split('.')[0]+'_patch_bg_s10'+sufix+str(i)+'.png'), bg)


# Extract N patches for self supervised learning - no label
def convert_one_image_self_sup(sel, resize_expand=True):
    """ Convert image to dicom and extract patches """

    global total_patches

    image_source = (os.path.join(TOP_VINDR, file_list[sel]['dir'], file_list[sel]['file']))+'.dicom'

    image_name = file_list[sel]['short_case_id']+'_'+file_list[sel]['side']+'_'+ \
                 file_list[sel]['view']+'_B'+file_list[sel]['birads'] + \
                 '_D'+file_list[sel]['density']+'.png'

    temp_image_path = os.path.join('./temp', image_name)
    dir_destiny_patch = save_path

    print(temp_image_path)

    # We must make same process as image: save converted dicom then patch from it.
    save_dicom_image_as_png(image_source, temp_image_path, file_list[sel]['side'] ,16)
    if RESIZE_EXPAND_CHANNELS:
        image = resize_expand_channels(temp_image_path, False)

    patch_size = PATCH_SIZE
    # vamos varrer imagem e pegar patches
    #  Overlap 50% (divide step by 2)
    #  Height is 1152, to use 5 patches we start from 16 and leave last 16 out
    step_width = patch_size // 2
    step_height = patch_size // 2

    valid_patches = 0
    patch_list = []

    y_steps = (DEFAULT_SIZE[1] // step_height) - 1
    x_steps = (DEFAULT_SIZE[0] // step_width) - 1

    for x in range(x_steps):
        for y in range(y_steps):

            patch = image[y*step_height:y*step_height+patch_size, x*step_width:x*step_width+patch_size]
            limiar, _ = cv2.threshold(patch, 0, image.max(), cv2.THRESH_OTSU)

            if debug:
                print(y, step_height, x, step_width, np.sum(patch)/(PATCH_SIZE*PATCH_SIZE), limiar)
                cv2.imshow('Real Patch', patch)
                cv2.waitKey(0)

            # Avoid all black areas
            if (limiar > 1):
                valid_patches += 1
                patch_list.append(patch)

    for i, p in enumerate(patch_list):
        cv2.imwrite(os.path.join(dir_destiny_patch, image_name.split('.')[0]+'_patch_self'+str(i)+'.png'), p)




#erro em a235482f7_R_MLO_B2_DC.


# # Versão original de processamento sequencial
# count2 = 0
# for image in selection:
#     # convert_one_image(image, RESIZE_EXPAND_CHANNELS)
#     convert_one_image_self_sup(image, RESIZE_EXPAND_CHANNELS)
#     count2 += 1
# print('Converted sequencial(all): ', count2)

# Fazendo por processamento paralelo - muito mais rápido

# Escolha o metodo, para SELF SL ou amostragem padrao Shen

f = partial(convert_one_image, resize_expand=RESIZE_EXPAND_CHANNELS)
# f = partial(convert_one_image_self_sup, resize_expand=RESIZE_EXPAND_CHANNELS)
p = Pool(NUM_PROCESSES)
p.map(f, selection)


print('Converted(all): ', count)
# print('Train count converted: ', count_train)