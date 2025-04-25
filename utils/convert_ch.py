# Convert channels from 3 to 1 for all images in directories below
import os
import cv2
import numpy as np
import sys

# assign directory
directory = './'
 
# iterate and save over files in that directory
img_list = []
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.png'):
            img_list.append(os.path.join(root, filename))

print('Will try to convert ', len(img_list), ' images.')

# img_list = [i for i in os.listdir()]
count = 0
for i, img in enumerate(img_list):
    image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    if len(image.shape) == 2:
        continue
    new_image = np.zeros((*image.shape[0:2], 1), dtype=np.uint16)
    new_image = image[:, :, 0]   # convert to one channel
    cv2.imwrite(img, new_image)
    print('.', end='', flush=True)
    count += 1
print('Converted ', count, ' images.')
