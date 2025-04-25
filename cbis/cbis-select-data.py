#
# Filter data from CBIS-DDSM
#
# Warning: this code is a patch from many other files, try to use function by function.
#          Use with caution
#          Used to prepare datasets for paper: https://ieeexplore.ieee.org/document/9837037
#        
#
# Separates images and corresponding masks according to a criteria HARDCODED HERE and input
# xls (mass or calcification).
#
# Read dicom, convert to PNG and crop exam image.
# Selects randomly a number of images. Save the images and masks with meaningfull dir name.
# Generate statistics for selection.
#
#  Ex.
#  python3 cbis-select-data.py -f '../csv/mass_case_description_train_set.csv' -n 10 -c './cropped2'
#
#  python3 cbis-select-data.py -f 'csv/mass_case_description_train_set.csv' -n 5 -c './mass_ben_repo'
#
#  python3 cbis-select-data.py -f 'csv/calc_case_description_test_set.csv' -n 119 
#                       -c '/media/dpetrini/DGPP02_1TB/usp/sinai_data/calc_test_benign_max'
#
#  python3 cbis-select-data.py -f './csv/calc_case_description_train_set.csv' -n 0 
#                     -o '/media/dpetrini/DGPP02_1TB/usp/sinai_data/cbis-ddsm-v2/calc_train_benign'
#
#
#  Obs. Para massas geralmente o tamanho de algumas imagens nao bate com as mascaras. Entao 
#       selecionar n maior e depois apagar algumas. Para calcificacoes nao tem esse problema
#       usar n desejado.
#
# DGPP - 12/10/2019 - Initial version for Unet studies (patches)
#      - 09/11/2019 - include support for masks. Read corresponding mask, crop and save 
#      - 29/11/2019 - Added couting of features like birads and image/mask size check (need action yet)
#      - 14/02/2020 - Using for selecting only malign/benign for Sinai implementation
#      - 15/07/2021 - Revisado para consertar bug de selecao de apenas abnormality id= e não ter BWC
#                     Agora pela o primeiroarquivo de cada exame não importando ab. id.

import pandas as pd
import numpy as np
import argparse
import sys
import random
from multiprocessing import Pool, Process

import png
import pydicom
import os
import cv2
import time

from crop_mammogram import crop_mammogram_one_image

# for multiprocess
num_processes = 2

DEFAULT_SIZE = (896, 1152)

RESIZE_EXPAND_CHANNELS = False
SKIP_IMG_GENERATION = True
SKIP_MASK_GENERATION = False

def save_dicom_image_as_png(dicom_filename, png_filename, bitdepth=12):
    """
    :param dicom_filename: path to input dicom file.
    Save 12-bit mammogram from dicom as rescaled 16-bit png file.
    :param png_filename: path to output png file.
    :param bitdepth: bit depth of the input image. Set it to 12 for 12-bit mammograms.
    """
    image = pydicom.read_file(dicom_filename).pixel_array
    with open(png_filename, 'wb') as f:
        writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=bitdepth, greyscale=True)
        writer.write(f, image.tolist())

# indicate here where dataset is
TOP_CBIS = '/media/dpetrini/DGPP02_1TB/Datasets/mammo/CBIS-DDSM/CBIS-DDSM'  #'CBIS-DDSM'
sufix = random.randint(1,200)
#TEMP_DIR = './img_full_'+str(sufix)
TEMP_DIR = '/media/dpetrini/DGPP02_1TB/usp/sinai_data/img_full_'+str(sufix)

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
    n_files = 5
    print('Using default number of files: 5')
else:
    n_files=int(args['nfiles'])

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
df_full = pd.read_csv(args['file'])

print ('Shape before:', df_full.shape)

# o certo é usar a linha abaixo para remover duplicados (exames com diversas
# segmentações) e manter o primeiro. Pois de inicio procuramos apenas linhas
# onde o Abnormality ID == 1 mas alguns exames comecam com Ab ID == 2, então
# perdemos esses. A linha abaixo remove os duplicados de uma maneira mais
# correta usando o nome da imagem como chave.

# Remove duplicates - it means we will select one entry per file as we only
#  want full images here, not patches or binary masks
df = df_full.drop_duplicates(subset=['image file path'])



# transforma em np para manipular
table = np.array(df)

print ('Shape depois:', table.shape)


file_list = []
mask_file_list = []

start_time = time.time()

# create vars for statistics
birads0 = 0;birads1 = 0;birads2 = 0;birads3 = 0;birads4 = 0;birads5 = 0
malignant=0;benign=0;benign_wcb=0
badcount=0

# Following lines filter what we want. Hardcoded now- -CRIE SUA SELECAO AQUI
#  get table columns from ** in EOF

############################# CODIGO ANTIGO - usado para gerar CBIS-DDSM até 2021-07-15 - BEGIN
# count = 0
# for i in range(table.shape[0]):
#     # seleciona birads 4 ou 5 & VIEW CC & abnormality ID=1, apenas 1° caso
#     #if ((table[i][8] == 4 or table[i][8] == 5) and (table[i][3] == 'CC') and table[i][4] == 1):
#     # selecionando CC e primeiro caso - todos birads.
#     #if ( ((table[i][3] == 'CC') or (table[i][3] == 'MLO')) and (table[i][4] == 1)):
#     # seleciona benign or malignant e primeiros casos
#     if table[i][9] == 'MALIGNANT' and table[i][4] == 1:
#     # if table[i][9] != 'MALIGNANT' and table[i][4] == 1:
#     # if table[i][9] == 'BENIGN' and table[i][4] == 1:
#     # if table[i][4] == 1:
#         #print(table[i][0], table[i][12])
#         file_list.append({'file': table[i][11], 'birads': table[i][8], 'pathology':table[i][9]})
#         #print('File: ', table[i][11])
#         mask_file_list.append(table[i][13])
#         count = count+1
#####################################################################3  CODIGO ANTIGO END


# CODIGO NOVO usando todos os arquivos e não precisamos mais filtrar pelo abnormal ID,
# basta filtrar pelo criterio base que desejamos
count = 0
for i in range(table.shape[0]):
    if abnormality == 'MALIGNANT':
        if table[i][9] == 'MALIGNANT':
            file_list.append({'file': table[i][11], 'birads': table[i][8], 'pathology': table[i][9]})
            count = count+1
    elif abnormality == 'BENIGN':
        if table[i][9] != 'MALIGNANT':
            file_list.append({'file': table[i][11], 'birads': table[i][8], 'pathology': table[i][9]})
            count = count+1

print('Current selection (all): ', count, len(file_list))

# if n_files == 0 then use all files
if n_files == 0:
    # print('Requested zero files, quiting...')
    # sys.exit(0)
    n_files = len(file_list)

if len(file_list) < n_files:
    print('Number of files requested:', n_files, 'bigger than available in current filter:', len(file_list))
    sys.exit()

# take a random selection of SIZE N
selection = random.sample(list(range(0,len(file_list))), n_files)
print ("Selection if N samples: ", selection)

# assert(len(file_list) == len(mask_file_list))

# sys.exit()


# pegar n imagens e copia para outro diretorio com nome obtido do path
# necessario fazer essa busca porque o caminho informado no csv nao eh correto 
print('Converting images to png: ', len(file_list))


def resize_expand_channels(img_file, expand=True):
    """ Resize to our default size and replicate channels to ease dataloaders """
    print(img_file)
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


def convert_one_image(sel, resize_expand=True):
    """ Convert image to dicom and prepare for dataloaders """
    image_file = file_list[sel]['file']
    first_dir = os.path.split(image_file)[0]
    first_dir = first_dir.split('/')[0]
    for curdir, second_dir, files in os.walk(os.path.join(TOP_CBIS, first_dir)):
        for f in files:
            if f.endswith('.dcm') and not f.startswith('.'):
                print('.', end=''); sys.stdout.flush()
                img_file = os.path.join(save_path, first_dir+'.png')
                save_dicom_image_as_png(os.path.join(curdir, f),
                                        img_file, 16)
                if resize_expand:
                    resize_expand_channels(img_file)

# parece ue nossa funcao nao esta preparada para multiprocess - muito loop ou syscall
# with Pool(num_processes) as pool:
#    pool.map(convert_one_image, selection)


# entao fazer serialmente:
if not SKIP_IMG_GENERATION:
    for image in selection:
        convert_one_image(image, RESIZE_EXPAND_CHANNELS)
else:
    print('Skipping image generation.')


# Agora tratamento para as máscaras, vamos usar todas entradas

file_list = []
mask_file_list = []
start_time = time.time()

# transforma em np para manipular
table_full = np.array(df_full)
print('\nShape depois:', table_full.shape)

count = 0
for i in range(table_full.shape[0]):
    if abnormality == 'MALIGNANT':
        if table_full[i][9] == 'MALIGNANT':
            mask_file_list.append(table_full[i][13])
            count = count+1
    # # if table_full[i][9] == 'MALIGNANT':
    # if table_full[i][9] != 'MALIGNANT':
    # if table[i][9] == 'BENIGN' and table[i][4] == 1:
    # if table[i][4] == 1:
        # file_list.append({'file': table[i][11], 'birads': table[i][8], 'pathology':table[i][9]})
        #print('File: ', table[i][11])

    elif abnormality == 'BENIGN':
        if table_full[i][9] != 'MALIGNANT':
            mask_file_list.append(table_full[i][13])
            count = count+1

if not SKIP_MASK_GENERATION:
    # do the same for masks
    print('\nConverting masks to png: ', len(mask_file_list))
    # for sel in selection:
    #     image_file = mask_file_list[sel]
    for image_file in mask_file_list:
        # image_file = mask_file_list[sel]
        first_dir = os.path.split(image_file)[0]
        #file_name = os.path.split(image_file)[1]
        first_dir = first_dir.split('/')[0]
        for curdir, second_dir, files in os.walk(os.path.join(TOP_CBIS, first_dir)):
            for f in files:
                if not f.startswith('.'): 
                    a = pydicom.dcmread(os.path.join(curdir, f))  # read dicom structures
                    if not hasattr(a,"BitsAllocated"): print("Erro: Nao tem campo BitsAllocated")
                    if a.BitsAllocated == 8:  # contem a mascara(8 bits), que queremos, e um patch da imagem (16bits)
                        print('.', end=''); sys.stdout.flush()
                        mask_file = os.path.join(save_path_mask, first_dir+'.png')
                        save_dicom_image_as_png(os.path.join(curdir, f),
                                                mask_file, a.BitsAllocated)
                        if RESIZE_EXPAND_CHANNELS:
                            resize_expand_channels(mask_file, False)

if not do_crop:
    print('No cropping requested, so leaving, done.')
    # print('Please erase manually not used dirs: ', cropped_file_path, cropped_mask_path)
    sys.exit()


# Image & Masks cropping
print('\nSaving images/masks: ')

remove_list = []

# Salva uma imagem
# Le a imagem, faz cropping e salva. Depois abre a mascara e tenta fazer o mesmo cropping na mascara
# se nao da certo apaga a imagem e a mascara. Isso parece ser um problema do CBIS-DDSM para algumas
# imagens conforme confirmamos para alguns casos. Mas pode ser um bug dessa funcao tambem.
def save_one_image(png_image):
    print('.', end=''); sys.stdout.flush()
    image = cv2.imread(os.path.join(save_path,png_image),-1)
    # crop, save and get crop coords for mask
    _, top, bottom, left, right = crop_mammogram_one_image(image, os.path.join(cropped_file_path,png_image))
    # reopen to read size
    img_check = cv2.imread(os.path.join(cropped_file_path,png_image),-1)

    # MASK - check if exit corresponding (with _1 or _2 from CBIS) and save cropped - now only take _1
    for n in range(1,2):
        if os.path.isfile(os.path.join(save_path_mask, png_image.split('.')[0]+'_'+str(n)+'.png')):
            mask = cv2.imread(os.path.join(save_path_mask, png_image.split('.')[0]+'_'+str(n)+'.png'), -1)
            mask = mask[top:bottom, left:right]
            cv2.imwrite(os.path.join(cropped_mask_path, png_image.split('.')[0]+'_'+str(n)+'.png'), mask)
            # reopen and check sizes
            mask_check = cv2.imread(os.path.join(cropped_mask_path, png_image.split('.')[0]+'_'+str(n)+'.png'),-1)
            # now take action if size mismatch
            if (img_check.shape[0:2] != mask_check.shape[0:2]):
                print('\n*** img - mask size dont match [Removing]: ', png_image, img_check.shape[0:2], mask_check.shape[0:2])
                

                #print(file_list[0]['file'])
                for sample in file_list:
                    apagueme = sample['file'].split('/')[0]
                    
                         
                    if apagueme.startswith(png_image[:29]):
                        #print(apagueme, png_image[:29], sample['birads'])
                        sample['birads'] = 1000
                        sample['pathology'] = 1000
                        # remove entao os arquivos recem criados
                        remove_list.append(os.path.join(cropped_file_path,png_image))
                        remove_list.append(os.path.join(cropped_mask_path, png_image.split('.')[0]+'_'+str(n)+'.png'))
                        #print(len(remove_list), remove_list)
                        #print('Removendo: ', os.path.join(cropped_file_path,png_image))
                        os.remove(os.path.join(cropped_file_path,png_image))
                        #print('Removendo mask: ', os.path.join(cropped_mask_path, png_image.split('.')[0]+'_'+str(n)+'.png'))
                        os.remove(os.path.join(cropped_mask_path, png_image.split('.')[0]+'_'+str(n)+'.png'))

def save_one_image_not_cropped(png_image):
    print('.', end=''); sys.stdout.flush()
    image = cv2.imread(os.path.join(save_path,png_image),-1)
    # crop, save and get crop coords for mask
    #_, top, bottom, left, right = crop_mammogram_one_image(image, os.path.join(cropped_file_path,png_image))
    top, bottom, left, right = 0, image.shape[0], 0, image.shape[1]
    print(image.shape, top, bottom, left, right)
    sys.exit()      ######### TESTAR
 
    # reopen to read size
    img_check = cv2.imread(os.path.join(cropped_file_path,png_image),-1)

    # MASK - check if exist corresponding (with _1 or _2 from CBIS) and save cropped - now only take _1
    for n in range(1,2):
        if os.path.isfile(os.path.join(save_path_mask, png_image.split('.')[0]+'_'+str(n)+'.png')):
            mask = cv2.imread(os.path.join(save_path_mask, png_image.split('.')[0]+'_'+str(n)+'.png'), -1)
            mask = mask[top:bottom, left:right]
            cv2.imwrite(os.path.join(cropped_mask_path, png_image.split('.')[0]+'_'+str(n)+'.png'), mask)
            # reopen and check sizes
            mask_check = cv2.imread(os.path.join(cropped_mask_path, png_image.split('.')[0]+'_'+str(n)+'.png'),-1)
            # now take action if size mismatch
            if (img_check.shape[0:2] != mask_check.shape[0:2]):
                print('\n*** img - mask size dont match [Removing]: ', png_image, img_check.shape[0:2], mask_check.shape[0:2])


                #print(file_list[0]['file'])
                for sample in file_list:
                    apagueme = sample['file'].split('/')[0]


                    if apagueme.startswith(png_image[:29]):
                        #print(apagueme, png_image[:29], sample['birads'])
                        sample['birads'] = 1000
                        sample['pathology'] = 1000
                        # remove entao os arquivos recem criados
                        remove_list.append(os.path.join(cropped_file_path,png_image))
                        remove_list.append(os.path.join(cropped_mask_path, png_image.split('.')[0]+'_'+str(n)+'.png'))
                        #print(len(remove_list), remove_list)
                        #print('Removendo: ', os.path.join(cropped_file_path,png_image))
                        os.remove(os.path.join(cropped_file_path,png_image))
                        #print('Removendo mask: ', os.path.join(cropped_mask_path, png_image.split('.')[0]+'_'+str(n)+'.png'))
                        os.remove(os.path.join(cropped_mask_path, png_image.split('.')[0]+'_'+str(n)+'.png'))



image_list = [image for image in os.listdir(save_path)]
for image in image_list:
    save_one_image(image)

# with Pool(num_processes) as pool:
#    pool.map(save_one_image, image_list)


# check stats
for sel in selection:
    item = file_list[sel]
    # count assesment type
    if item['birads'] == 0: birads0 +=1
    elif item['birads'] == 1: birads1 +=1
    elif item['birads'] == 2: birads2 +=1
    elif item['birads'] == 3: birads3 +=1
    elif item['birads'] == 4: birads4 +=1
    elif item['birads'] == 5: birads5 +=1
    elif item['birads'] == 1000: badcount +=1
    # count pathology type
    if item['pathology'] == 'MALIGNANT': malignant +=1
    elif item['pathology'] == 'BENIGN': benign +=1
    elif item['pathology'] == 'BENIGN_WITHOUT_CALLBACK': benign_wcb +=1  


# remove bad files (differente sizes for mask & image shape)
# print(len(remove_list), remove_list)
# for file in remove_list:
#     print('Removendo: ', file)
#     os.remove(file)


#sys.exit()

# Remove intermediate png images
print('\nDeleting temp dirs...')
for curdir, dirs, files in os.walk(save_path, topdown=False): 
    for name in files: 
        os.remove(os.path.join(curdir, name))
os.removedirs(save_path) 

for curdir, dirs, files in os.walk(save_path_mask, topdown=False): 
    for name in files: 
        os.remove(os.path.join(curdir, name))
os.removedirs(save_path_mask) 

# final check
assert(len(os.listdir(cropped_file_path)) == len(os.listdir(cropped_mask_path)))

print('Birads count (0 to 5):')
print(birads0, birads1, birads2, birads3, birads4, birads5)

print('Pathology count (malignant, benign, benign_wcb) :')
print(malignant, benign, benign_wcb)

print('Total: ', birads0+birads1+birads2+birads3+birads4+birads5, malignant+benign+benign_wcb)
print('Bad size matches: ', badcount)

print('Elapsed time: {:.2f} s.'.format(time.time()-start_time))


# **
# Data columns (total 14 columns):
# patient_id                 378 non-null object
# breast_density             378 non-null int64
# left or right breast       378 non-null object
# image view                 378 non-null object
# abnormality id             378 non-null int64
# abnormality type           378 non-null object
# mass shape                 378 non-null object
# mass margins               361 non-null object
# assessment                 378 non-null int64
# pathology                  378 non-null object
# subtlety                   378 non-null int64
# image file path            378 non-null object
# cropped image file path    378 non-null object
# ROI mask file path         378 non-null object


