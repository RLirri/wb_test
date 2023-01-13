import cv2
import numpy as np


if __name__ == '__main__':

    img = cv2.imread('../pic/Fig0304(a)(breast_digital_Xray).tif', 0)

    """
  Canny can only process grayscale images, so it converts the read images into grayscale images
    Gaussian smoothing is used to denoise the original image
    Call canny function to specify the maximum and minimum thresholds, where aperturesize is 3 by default
    """
    img = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(img, 50, 150)

    cv2.imshow('origin', img)
    cv2.imshow('Canny', canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
