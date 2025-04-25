# Count how many of CBIS-DDSM files are from same exam and side.
#
# Ex. Calc-Test_P_01562_LEFT_CC.png AND Calc-Test_P_01562_LEFT_MLO.png
#
# Execution: python3 dual_view_separe.py 'train/benign'
#
# PEGA A CATEGORIA como o ultimo diretorio do path
# cria dicionario
#
# DGPP 2020-08-28

import sys
import os


def main():

    path = './'
    if len(sys.argv) == 2:
        path = sys.argv[1]
    elif len(sys.argv) > 2:
        print('Error, please insert only the folder to be swept.')
        sys.exit()

    file_list = [i for i in os.listdir(path) if i.endswith('.png')]
    file_list.sort()
    total_files = len(file_list)
    count = 0
    category = path.split('/')[-2]
    exam = []

    for j in range(total_files):

        if file_list[j][:7] == 'Calc-Tr' or file_list[j][:7] == 'Mass-Tr':
            sub = 16
        elif file_list[j][:7] == 'Calc-Te' or file_list[j][:7] == 'Mass-Te':
            sub = 12

        if (j+1) != total_files:
            if file_list[j][:sub+5] == file_list[j+1][:sub+5]:
                # check if have same SIDE
                if file_list[j][sub+5: sub+5+4] != file_list[j+1][sub+5: sub+5+4]:
                    continue
                case = file_list[j][sub: sub+5]  # case
                if file_list[j][sub+6: sub+7] == 'R':
                    side = 'right'
                elif file_list[j][sub+6: sub+7] == 'L':
                    side = 'left'
                else:
                    print('Wrong image file name from dataset ', os.path.basename(__file__))
                    sys.exit
                cc_file = file_list[j]
                mlo_file = file_list[j+1]

                exam.append({
                    'case': case,
                    'CC': cc_file,
                    'MLO': mlo_file,
                    'side': side,
                    'label': category
                })

                count += 1

    for case in exam:
        print(case)
    print('Ocorrencias: ', count)


if __name__ == '__main__':
    main()
