import cv2
import math
import numpy as np
from builtins import range, print


def logTransform(c, image):


    #GreyScale
    h, w = img.shape[0], img.shape[1]
    new_img = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            new_img[i, j] = c * (math.log(1.0 + img[i, j]))

    new_img = cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)/255.

    return new_img


if __name__ == '__main__':
    img = cv2.imread('../pic/Fig0305(a)(DFT_no_log).tif', 0)
    log_img = logTransform(1, img)
    cv2.imshow('src_img', img)
    cv2.imshow('log_img', log_img)
    cv2.waitKey(0)
