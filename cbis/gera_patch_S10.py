#
# Gera patches (image+mask)  positivos centrados no positivo da mascara
#
# Estamos pegando mascaras 224x224 diretamente. Se W ou H do patch original
# sao menores que 224 desprezamos a sobra. Pode ser que isso influencie.
# Se precisar pega patch da maior dimensao e fazer resize no final para 224.
#
# Patches para Sinai tipo S10
#
#  Obs.:
#  - Diretorio de imagens hardcoded no fim desse arquivo.
#
# Ex.
# python3 gera_patch_S10.py -i \
#    '/media/dpetrini/DGPP02_1TB/usp/sinai_data/cbis-ddsm-separado-4-cat-full/calc_malign'
#
# 22/03/2020 - naiores tamanhos de patch para recorte final na augmentation
# 20/07/2021 - atualizado para gerar patches cbis v2 (Full)


import os
import cv2
import numpy as np
import sys
import random
import argparse
from random import randint

PATCH_SIZE = 224        # Usar BORDER_REFLECT na aug. para remover pretos

# using accoding to paper, also to fit better in GPU
height_s = 1152
width_s = 896

# limite superior de pixels pretos para background
NON_BLACK_LIMIT = 0  # 1000 - mudar para zero. Paper nao menciona nada.


def preparaImage(imagePath, dest_path, maskPath, dest_mask_path, dest_bg_patch, patch_size):

    over_limit_x = 0
    over_limit_y = 0

    if os.path.isdir(dest_path) is False:
        print('Creating ', dest_path)
        os.makedirs(dest_path, exist_ok=False)

    if os.path.isdir(dest_mask_path) is False:
        print('Creating ', dest_mask_path)
        os.makedirs(dest_mask_path, exist_ok=False)

    if os.path.isdir(dest_bg_patch) is False:
        print('Creating ', dest_bg_patch)
        os.makedirs(dest_bg_patch, exist_ok=False)

    # erase content in destination folders
    print('Deleting files in destination folders')
    for curdir, dirs, files in os.walk(dest_path, topdown=False):
        for name in files:
            os.remove(os.path.join(curdir, name))

    for curdir, dirs, files in os.walk(dest_mask_path, topdown=False):
        for name in files:
            os.remove(os.path.join(curdir, name))

    for curdir, dirs, files in os.walk(dest_bg_patch, topdown=False):
        for name in files:
            os.remove(os.path.join(curdir, name))

    imageList = [f for f in os.listdir(imagePath) if os.path.isfile(os.path.join(imagePath, f))]
    mask_list = [f for f in os.listdir(maskPath) if os.path.isfile(os.path.join(maskPath, f))]
    imageList.sort()
    mask_list.sort()

    print('Qty images: ', len(imageList), len(mask_list))
    # assert (len(imageList) == len(mask_list))

    n = len(imageList)

    count_new_patches = 0
    count_multiple = 0
    count_patches = 0


    for n in range(len(imageList)):

        # Read image already converted to 896x1152 (sinai)    
        img = cv2.imread(os.path.join(imagePath, imageList[n]), -1)

        # generate a list of all possible abnormalities for this image
        abnormality_list = [l for l in mask_list if l.startswith(imageList[n].split('.')[0])]
        n_abnormality = len(abnormality_list)

        h_img, w_img = img.shape[0:2]

        # will store forbidden areas for each lesion
        mask_bg = np.zeros((img.shape[0:2]), dtype=np.uint8)

        for i, mask_file in enumerate(abnormality_list):

            count_patches += 1

            print('-> ',  imageList[n], mask_file, count_patches)
            
            mask = cv2.imread(os.path.join(maskPath,  mask_file), 0)

            # pegar bounding box do comp con
            connectivity = 4
            # Perform the operation
            output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
            # The first cell is the number of labels
            num_labels = output[0]
            # The second cell is the label matrix
            labels = output[1]
            # The third cell is the stat matrix
            stats = output[2]
            # The fourth cell is the centroid matrix
            centroids = output[3]

            x, y, w, h = stats[1][0], stats[1][1], stats[1][2], stats[1][3]

            # VAMOS EXLUIR OS EXAMES COM MAIS DE UMA LESAO!!!!!!
            # (provavelmente deprecado - pois pegamos cada anormalidade da imagem - 2021)
            if num_labels != 2:
                print('-----> Excluindo imagem com mais de uma lesao - Num: ', num_labels-1, mask_file)
                count_new_patches += (num_labels - 2)
                count_multiple += 1
                continue

            # get center
            x_center = x + w//2
            y_center = y + h//2

            # Overlap = minimum 90% so skew is 10% * patch size
            skew = patch_size//10

            # consider x,y from center and check limits w/ (1-overlap)
            if (x_center + patch_size//2 + skew) > w_img:
                x = w_img - patch_size - skew - 1
            elif (x_center - patch_size//2 - skew) > 0:
                x = x_center - patch_size//2
            else:
                x = 0 + skew
                over_limit_x += 1

            if (y_center + patch_size//2 + skew) > h_img:
                y = h_img - patch_size - skew - 1
            elif (y_center - patch_size//2 - skew) > 0:
                y = y_center - patch_size//2
            else:
                y = 0 + skew
                over_limit_y += 1

            # x = x_center - patch_size//2
            # y = y_center - patch_size//2

            for i in range(10):
                # Pega overlap de pelo menos 90%: desloca até 10% somado
                # para cada lado de x ou y - soma tem que dar skew
                x_skew = randint(-skew, skew)
                y_skew = (skew - x_skew) if x_skew > 0 else -(skew - abs(x_skew))
                xx = x + x_skew
                yy = y + y_skew

                # make patches and save
                sub_mask = mask[yy:yy+patch_size, xx:xx + patch_size]
                sub = img[yy:yy+patch_size, xx:xx + patch_size]

                if (0):
                    print(i, xx, yy, skew, x_skew, y_skew)
                    cv2.imshow('mask', sub_mask)
                    cv2.imshow('sub', sub)
                    cv2.waitKey(0)

                cv2.imwrite(os.path.join(dest_path, mask_file.split('.')[0]+'_patch_s10_'+str(i)+'.png'), sub)
                cv2.imwrite(os.path.join(dest_mask_path, mask_file.split('.')[0]+'_mask_patch_s10_'+str(i)+'.png'), sub_mask)

            # Let's protect positive area by painting white as forbidden areas.
            # Background patches will not be sampled from here.
            mask_bg[y-skew:y+patch_size+skew, x-skew:x+patch_size+skew] = 255

        if False:
            cv2.imshow('mask safe', mask_bg)
            cv2.waitKey(0)

        # obtem n_bg_patches patches randomicos negativos (background)
        # print("--> Now getting negative patches (background) x", n_abnormality)

        n_bg_patches = 10 * n_abnormality
        valid_patches = 0
        while (True):
            x = randint(0, w_img-patch_size-1)
            y = randint(0, h_img-patch_size-1)
            sub      = img [y:y+patch_size, x:x+patch_size]
            sub_mask = mask[y:y+patch_size, x:x+patch_size]  # should be mask_bg, but not so wrong being mask only - will not overlap lesion

            # Dont't patch from safe area and avoid all black areas
            if (np.sum(sub_mask) == 0 and np.sum(sub) > NON_BLACK_LIMIT):
                valid_patches += 1
                cv2.imwrite(os.path.join(dest_bg_patch, imageList[n].split('.')[0]+'_patch_bg_s10_'+str(valid_patches)+'.png'), sub)
                if valid_patches == n_bg_patches:
                    break

    # numero de patches que ficaram nas bordas, correm risco de nao
    # ... acabou nao tendo muita importancia
    print('Patches que atingiram limite (x, y) ', over_limit_x, over_limit_y)
    print('Abandoned patches considering multiple lesions in single image: ',
          count_new_patches, ' Imagens count: ', count_multiple)

    return


# MAIN
ap = argparse.ArgumentParser(description='Create patches S10 224x224 for image/mask centered in lesions and additional bgs.')
ap.add_argument("-i", "--input",  required=True,
                help="folfder to generate patches from. (no trailing /")
ap.add_argument("-o", "--output", required=False,
                help="root folder to store patches")

args = vars(ap.parse_args())

input_path = args['input']
input_folder = input_path.split('/')[-1]

if args['output'] is None:
    output_path = 'prepare_data/patches_S10/'
    print('Using default path: prepare_data/patches_S10/',
          os.path.join(output_path, input_folder + '_S10'))
else:
    output_path = args['output']

# create folders for masks
ground_truth = input_path + "_mask"
ground_truth_folder = ground_truth.split('/')[-1]

# call main function
preparaImage(input_path, os.path.join(output_path, input_folder+'_S10'),
             ground_truth, os.path.join(output_path, ground_truth_folder+'_S10'),
             os.path.join(output_path, input_folder+'_bg_S10'),
             PATCH_SIZE)
