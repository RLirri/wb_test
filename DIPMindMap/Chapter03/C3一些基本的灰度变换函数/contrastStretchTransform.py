import math
from builtins import range, print
import numpy as np
import cv2


def contrastStretchTransform(image):

    h, w, d = image.shape[0], image.shape[1], image.shape[2]
    new_img = np.zeros((h, w, d), dtype=np.float32)
    A = image.min()
    B = image.max()
    print(A, B)
    for i in range(h):
        for j in range(w):
            for k in range(d):
                new_img[i, j, k] = 255.0 / (B - A) * (image[i, j, k] - A) + 0.5
    new_img = cv2.normalize(new_img, new_img, 0, 255, cv2.NORM_MINMAX)
    new_img = cv2.convertScaleAbs(new_img)

    return new_img


if __name__ == '__main__':
    img = cv2.imread('../pic/beizi.png', 1)
    contrast_img = contrastStretchTransform(img)
    cv2.imshow('src img', img)
    cv2.imshow('contrast img', contrast_img)
    cv2.waitKey(0)

