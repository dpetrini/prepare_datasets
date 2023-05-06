import numpy as np
import cv2

DEBUG = False

def data_preprocess(data, side):

    image = data.pixel_array

    depth_allocated = data.data_element('BitsAllocated').value
    depth_stored = data.data_element('BitsStored').value
    photometricInterpretation = data.data_element('PhotometricInterpretation').value

    if depth_allocated != 16 or depth_stored not in [12, 14, 16]:
        print('Invalid depth: ', depth_allocated, depth_stored)

    # Monochrome1: minimum=white, so need to invert
    # https://dicom.innolitics.com/ciods/rt-dose/image-pixel/00280004
    if str(photometricInterpretation) == 'MONOCHROME1':
        image = np.invert(image)

    if DEBUG:
        print(depth_allocated, depth_stored, image.shape)

    min = image.min()
    max = image.max()

    # Remove marca L-CC, etc sempre no canto superior contrario ao lado
    LIMIT_Y = int(0.12*(image.shape[0]))
    LIMIT_X = int(0.28*(image.shape[1]))

    # Seleciona o lado
    if side == 'R':
        image[0:LIMIT_Y, 0:LIMIT_X] = min
    elif side == 'L':
        image[0:LIMIT_Y, (image.shape[1] - LIMIT_X):image.shape[1]] = min
    else:
        pass

    # Encontra limiar OTSU
    limiar, _ = cv2.threshold(image, 0, max, cv2.THRESH_OTSU)
    # troca o minimo da imagem para limiar, para melhorar a normal da imagem
    image[image < limiar] = limiar

    # incluir processamento de 4*sigmas para extrair limites extremos
    image_flat = image.flatten()
    n_sigma = 4

    # Remove the limiar
    rm = np.array([limiar])
    # np.in1d return true if the element of `a` is in `rm`
    idx = np.in1d(image_flat, rm)
    clean_array = image_flat[~idx]

    # Obtain normal parameters and find the limits based in 4*sigma
    mu, sigma = np.mean(clean_array), np.std(clean_array)
    limite_inf = int(mu-n_sigma*sigma)
    limite_sup = int(mu+n_sigma*sigma)

    image[image < limite_inf] = limite_inf
    image[image > limite_sup] = limite_sup

    return image, depth_stored