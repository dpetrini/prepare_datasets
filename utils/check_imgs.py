# Chack if images are valid in folder
import os
import cv2
import numpy as np
import sys

# assign directory
directory = './500k'
 
# iterate and save over files in that directory
img_list = []
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.png'):
            img_list.append(os.path.join(root, filename))

print('Will try to check ', len(img_list), ' images.')


count = 0
issues = 0
for i, f in enumerate(img_list):
    # print(f)
    image = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    count = i
    if (i % 100) == 0:
    	print('\r', i, end='')
    try:
        if (len(image.shape)) > 3:
            print("error")
    except:
        print(f)
        issues += 1

print(f'Checked {count} files, found {issues} issues.')
