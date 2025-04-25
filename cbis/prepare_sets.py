#
# Prepara conjuntos de train/val/test para um diretorio de imagens inteiro
# pode ser usado para destinar varios inputs para a mesma saida (um por um)
#
# Ex.
# python3 prepare_sets.py -i '/media/dpetrini/DGPP02_1TB/usp/sinai_data/cbis-ddsm-separado-2-cat/benign' -o './temp' -t 76 -v 9
#

import argparse
import random
from shutil import copyfile
import os
import math

# MAIN
ap = argparse.ArgumentParser(description='Create train/va/test folders for input folder.')
ap.add_argument("-i", "--input",  required=True, help="input dir.")
ap.add_argument("-o", "--output", required=True, help="output for train/val/test")
ap.add_argument("-t", "--train",  required=True, help="train %.")
ap.add_argument("-v", "--val",  required=True, help="val%.")

args = vars(ap.parse_args())

input_dir = args['input']
output_dir = args['output']
train_size = int(args['train'])
val_size = int(args['val'])

# cria diretorios de saida
if os.path.isdir(os.path.join(output_dir, 'train')) is False:
    os.makedirs(os.path.join(output_dir, 'train'))
if os.path.isdir(os.path.join(output_dir, 'val')) is False:
    os.makedirs(os.path.join(output_dir, 'val'))
if os.path.isdir(os.path.join(output_dir, 'test')) is False:
    os.makedirs(os.path.join(output_dir, 'test'))

# cria lista de arquivos imagem
listdir = os.listdir(input_dir)
input_image_list = [file for file in listdir if file.endswith('png')]

# shuffle file list
random.shuffle(input_image_list)

train_size = math.floor((len(input_image_list) * train_size) / 100)
val_size = math.floor((len(input_image_list) * val_size) / 100)
test_size = math.floor(len(input_image_list) - train_size - val_size)

# partition
train = input_image_list[0: train_size]
val = input_image_list[train_size: train_size+val_size]
test = input_image_list[train_size+val_size:train_size+val_size+test_size+1]

# copia os arquivos
for file in train:
    copyfile(os.path.join(input_dir, file),
             os.path.join(output_dir, 'train', file))
for file in val:
    copyfile(os.path.join(input_dir, file),
             os.path.join(output_dir, 'val', file))
for file in test:
    copyfile(os.path.join(input_dir, file),
             os.path.join(output_dir, 'test', file))
