# 
#  Process patch - type S10 - Good for CBIS-DDSM, VINDR-MAMMO, so far.
#
#  Extract patches from mammography images, reading center of mass of lesions.
#  Extract 10 patches (and binary mask if available) around center plus 10
#  background patches (from anywhere else but patch).
#
#  DGPP
#
#  Date: 2020~2022


import cv2
import numpy as np
from random import randint

DEFAULT_SIZE = (896, 1152)


def extract_one_patches(img, x, y, w, h, ps=224, debug=False):

    height = DEFAULT_SIZE[1]
    width = DEFAULT_SIZE[0]

    patch = np.zeros((ps, ps), np.uint16)

    # fix lesion near border
    filled_image_x = width - x
    filled_image_y = height - y

    if filled_image_y < ps:
        print('\nY value bad, skipping\n', y, filled_image_y)
        return [0]

    if filled_image_x < ps:
        print('.')
        patch[:, 0:filled_image_x] = img[y:y+ps, x:x+ps]
    else:
        patch = img[y:y+ps, x:x+ps]

    
    limiar, _ = cv2.threshold(patch, 0, img.max(), cv2.THRESH_OTSU)

    if patch.shape[0] != ps or patch.shape[1] != ps:
        print('Wrong size PATCH, maybe border, skipping ', patch.shape, x, y)
        return [0]

    if w > ps or h > ps:
        print('\t Sizes bigger than patch size w, h, ps: ', w, h, ps)

    if debug:
        print(y, x, ps, np.sum(patch)/(ps*ps), limiar)
        cv2.imshow('Real Patch', patch)
        cv2.waitKey(0)

    return patch

    # Dont't patch from safe area and avoid all black areas
    if (limiar > 1):
        return patch
    else:
        print('\t Below Limiar Patch!')
        return [0]


def extract_s10_patches(img, mask, x, y, w, h, N=10, ps=224, debug=False):
    """
    Extract N patches from img around center +-10% with ps dimensions.

    Args:
        img: the full image in final resolution
        mask: the binary mask of patch in same resolution as image (optional)
        x,y: left bottom coordinates of patch im image.
        w,h: real width, heith from patch.
        N: number of lesion patches to extract.
        ps: size of patch to extract
    Returns:
        patch_list: list of N extracted patches
        mask_list: list of N binary mask of extracted patches
                   None if no input binary mask
        mask_bg: binary mask containing annotation of extracted patches
    """
    # get center
    x_center = x + w//2
    y_center = y + h//2

    over_limit_x = 0
    over_limit_y = 0

    h_img, w_img = img.shape[0:2]
    mask_bg = np.zeros((img.shape[0:2]), dtype=np.uint8)

    # Overlap = minimum 90% so skew is 10% * patch size
    skew = ps//10

    # consider x,y from center and check limits w/ (1-overlap)
    if (x_center + ps//2 + skew) > w_img:
        x = w_img - ps - skew - 1
    elif (x_center - ps//2 - skew) > 0:
        x = x_center - ps//2
    else:
        x = 0 + skew
        over_limit_x += 1

    if (y_center + ps//2 + skew) > h_img:
        y = h_img - ps - skew - 1
    elif (y_center - ps//2 - skew) > 0:
        y = y_center - ps//2
    else:
        y = 0 + skew
        over_limit_y += 1

    patch_list = []
    mask_list = []

    for i in range(N):
        # Pega overlap de pelo menos 90%: desloca até 10% somado
        # para cada lado de x ou y - soma tem que dar skew
        x_skew = randint(-skew, skew)
        y_skew = (skew - x_skew) if x_skew > 0 else -(skew - abs(x_skew))
        xx = x + x_skew
        yy = y + y_skew

        # make patches and save
        sub = img[yy:yy+ps, xx:xx + ps]
        if type(mask) is np.ndarray: sub_mask = mask[yy:yy+ps, xx:xx + ps]

        if (debug):
            print(i, xx, yy, skew, x_skew, y_skew)
            if type(mask) is np.ndarray: cv2.imshow('mask', sub_mask)
            cv2.imshow('S10 patch', sub)
            cv2.waitKey(0)

        patch_list.append(sub)
        if type(mask) is np.ndarray: mask_list.append(sub_mask)

    # Let's protect positive area by painting white as forbidden areas.
    # Background patches will not be sampled from here.
    mask_bg[y-skew:y+ps+skew, x-skew:x+ps+skew] = 255

    if debug:
        cv2.imshow('mask safe', mask_bg)
        cv2.waitKey(0)

    return patch_list, mask_list, mask_bg
    

def extract_bg_patches(img, mask, N=10, ps=224, limit=0):
    """
    Extract N random patches from img background. 
    Excluding area used for lesion patches and total black areas.

    Args:
        img: the full image in final resolution
        mask: the binary mask containing "forbidden" region
        N: number of lesion patches to extract.
        ps: size of patch to extract.
        limit: a minimum number of pixels to consider region not all black
    Returns:
        patch_list: list of N extracted patches
        mask_list: list of N binary mask of extracted patches
                   None if no input binary mask
        mask_bg: binary mask containing annotation of extracted patches
    """
    h_img, w_img = img.shape[0:2]
    n_bg_patches = N
    valid_patches = 0
    patch_list = []

    # keep trying to patch from image considering conditions
    while (True):
        x = randint(0, w_img-ps-1)
        y = randint(0, h_img-ps-1)
        bg = img[y:y+ps, x:x+ps]
        bg_mask = mask[y:y+ps, x:x+ps]
        # Dont't patch from safe area and avoid all black areas
        if (np.sum(bg_mask) == 0 and np.sum(bg) > limit):
            valid_patches += 1
            patch_list.append(bg)
            if valid_patches == n_bg_patches:
                break
    return patch_list
