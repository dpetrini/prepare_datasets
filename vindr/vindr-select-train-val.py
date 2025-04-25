#
# Filter data from VINDR-MAMMO
#
# Separates Train and validation data from already separated train.
#
# usa os arquivos gerados pelo notebook dividindo train-val 90-10
#
#
# PETRINI - 2022-09-04

# Soh rodou no python cmdline, arquivos movidos com sucesso.

# devemos rodar no dir: prepare_data/csv/vindr-mammo que contem os csvs

import pandas as pd
import os
import shutil

# criar os diretorios val:
os.makedirs('../../data-vindr/val/benign')
os.makedirs('../../data-vindr/val/malign')

# ir ao diretorio do csv e abrir terminal

# vamos abrir o csv preparado pelo notebook, que contem a coluna validation balanceada.
df = pd.read_csv('./breast-level_annotations_val.csv')

# Vamos criar um novo dataframe com base na coluna validation do csv
df_val = df.query("fold == 'validation'")

# linha abaixo para analisar apenas
for index, row in df_val.iterrows():
	print(index, row['study_id'][0:10])

# agrupar por paciente
df_val_clean = df_val.groupby('study_id').count()     # deve selecionar 401 linhas

# apenas para confirmar o resultado
for index, row in df_val_clean.iterrows():			  # deve mostrar os IDs dos pacientes (folders)
	print(index)
	
DESTINY_DIR = '../../data-vindr/val/benign'
SOURCE_DIR = '../../data-vindr/train/benign'

lista = os.listdir(SOURCE_DIR)

count = 0
for index, row in df_val_clean.iterrows():
	#print(index[0:10])
	for file in lista:
		if index[0:9] == file[0:9]:
			print('Movendo: ', file)
			count += 1
			if os.path.isfile(os.path.join(SOURCE_DIR, file)):
				shutil.move(os.path.join(SOURCE_DIR, file), os.path.join(DESTINY_DIR, file))
print(count)	# 1450


DESTINY_DIR = '../../data-vindr/val/malign'
SOURCE_DIR = '../../data-vindr/train/malign'

lista = os.listdir(SOURCE_DIR)

# Repetir loop acima

print(count)	# 154

# OBs: o mesmo vale para os patches, troca o CSV e faz para os diretorios malign_patch e _bg

# SOURCE_DIR = '/media/dpetrini/PETRINI/TPU/patches/train/malign_bg'
# DESTINY_DIR = '/media/dpetrini/PETRINI/TPU/patches/val/malign_bg'
# print(count) 1650