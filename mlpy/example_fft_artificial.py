"""
-------------------------------------------------------------------------
Example script for the Fourier transform usage (BCs)
-------------------------------------------------------------------------
This is an example script for differences between symmetric BCs and smoothed
or finite dots in artificial generated images.

.. moduleauthor:: Michael Barbier <michael@unam.bilkent.edu.tr>
"""


# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, filters

# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------
def generate_lattice_square(a, n, dot_radius, shift_x, shift_y):
    """
    Generates an image with dots on a period square lattice with lattice parameter a
    and dot size given by the dot_radius.

    :param a: The lattice length parameter
    :param n: Number of the repeating the basic lattice in both x and y direction
    :param dot_radius: radius of the plotted dots (should be non-negative < a/2), if zero, single pixels are used
    :param shift_y: Shift the dot in x direction (should be non-negative < a)
    :param shift_x: Shift the dot in y direction (should be non-negative < a)
    :return: The generated img
    """
    sz = a * n
    shape = (sz, sz)

    img = np.zeros(shape, np.dtype('float'))
    for ix in range(0, int(sz/a)):
        for iy in range(0, int(sz/a)):
            if dot_radius == 0:
                img[ix*a+shift_x, iy*a+shift_y] = 1.0
            else:
                cv.circle(img, (ix*a+shift_x, iy*a+shift_y), radius=dot_radius, color=1, thickness=cv.FILLED)
    return img


def show_fft(img):
    """
    Shows the Fourier spectrum together with the original image.
    The phase, magnitude and the magnitude in decibel is shown.
    Remark that the log of the magnitude (in db) is not defined if it is zero (resulting in wrong white pixels)

    :param img: input image
    :return: Nothing
    """
    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    magnitude_spectrum_db = 20 * np.log(np.abs(fshift))
    phase_spectrum = np.angle(fshift)

    plt.subplot(222), plt.imshow(phase_spectrum, cmap='gray')
    plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(223), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.subplot(224), plt.imshow(magnitude_spectrum_db, cmap='gray')
    plt.title('Magnitude Spectrum (in dB)'), plt.xticks([]), plt.yticks([])
    plt.show()


# -------------------------------------------------------------------------
# Script
# -------------------------------------------------------------------------

# At a region of square lattice
x0 = 0
y0 = 0
a = 16
n = 4
sz = a*n
size_crop = sz-1
shift_x = round(a/2)
shift_y = round(a/2)

# show the original image with Fourier spectra
dot_radius = 0
img_ori = generate_lattice_square(a, n, dot_radius, shift_x, shift_y)
show_fft(img_ori)

# show the image with asymmetric BCs due to crop: the Fourier spectrum is not perfect
img_asymmetric_1 = img_ori[y0:y0+sz-1, x0:x0+sz-1]
img_asymmetric_2 = img_ori[y0:y0+sz-2, x0:x0+sz-2]
img_asymmetric_3 = img_ori[y0:y0+sz-3, x0:x0+sz-3]
show_fft(img_asymmetric_1)
show_fft(img_asymmetric_2)
show_fft(img_asymmetric_3)

# Use of finite size dots instead of single pixel dots
dot_radius = 3
img_ori_4 = generate_lattice_square(a, n, dot_radius, shift_x, shift_y)
show_fft(img_ori_4)

# Smooth the image with a Gaussian
sigma = 2
img_smooth = filters.gaussian(img_ori_4, sigma=sigma)
show_fft(img_smooth)
