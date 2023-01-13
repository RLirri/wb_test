from builtins import print, int
import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('../pic/Fig0438(a)(bld_600by600).tif', 0)
fft = np.fft.fft2(img)
print("fft is: ", fft)
fftshift = np.fft.fftshift(fft)
print("fftshift is: ", fftshift)
magnitude_spectrum = 20 * np.log(np.abs(fftshift))
print(magnitude_spectrum[300, 300])

rows, cols = img.shape
crow, ccol = rows / 2, cols / 2

fftshift[int(crow - 30):int(crow+30), int(ccol-30):int(crow+30)] = 0
magnitude_spectrum_filter = 20 * np.log(np.abs(fftshift)+0.0000001)

f_ishift = np.fft.ifftshift(fftshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('origin image')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('magnitude_spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(magnitude_spectrum_filter, cmap='gray')
plt.title('High Pass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back, cmap='gray')
plt.title("High Pass Result"), plt.xticks([]), plt.yticks([])
plt.show()
