"""
-------------------------------------------------------------------------
Example script for the Fourier transform usage
-------------------------------------------------------------------------
This is an example script for using the fast Fourier transform on image crops
The goal eventually is to extract some lattice parameters from the spectral image (not shown here).

.. moduleauthor:: Michael Barbier <michael@unam.bilkent.edu.tr>
"""


# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------
def show_fft(img):
    img_crop = img[y0:y0+size_crop, x0:x0+size_crop]
    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_crop, cmap='gray')
    plt.title('Cropped image'), plt.xticks([]), plt.yticks([])

    f = np.fft.fft2(img_crop)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    plt.subplot(133), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


# -------------------------------------------------------------------------
# Script
# -------------------------------------------------------------------------

# At a region of square lattice
img = cv.imread('../data/square_1.png', 0)
x0 = 260
y0 = 360
size_crop = 80
show_fft(img)

# Not at a region of interest
img = cv.imread('../data/square_1.png', 0)
x0 = 460
y0 = 460
size_crop = 80
show_fft(img)

# At a region of quasi-crystal lattice
img = cv.imread('../data/quasi_1.png', 0)
x0 = 180
y0 = 180
size_crop = 130
show_fft(img)

# At a region of quasi-crystal lattice but smaller crop-size (the fourier spectrum has also less resolution now)
img = cv.imread('../data/quasi_1.png', 0)
x0 = 200
y0 = 200
size_crop = 60
show_fft(img)
