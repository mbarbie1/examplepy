"""
-------------------------------------------------------------------------
Module detection
-------------------------------------------------------------------------
Module for particle/spot detection in an image/video

.. module:: detection
    :platform:
    :synopsis: Module for particle/spot detection in an image/video

.. moduleauthor:: Michael Barbier <michael@unam.bilkent.edu.tr>
"""

# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import sys
import os
import numpy as np
import cv2 as cv
from skimage import io, filters, measure, color, exposure
import sys
import ffmpeg
import json

print(cv.__version__)
n_comment_dash = 80
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------


def detection_cht(im, radius):
    """
    Detect particles using the Circular Hough Transform (CHT)

    :param im: The input image which should be grey-valued
    :param radius: The radius of the particles used by the CHT algorithm
    :return: [1,2]:
        (1) A list with [x, y, radius] values,
        (2) The smoothed image used as input to the CHT algorithm
    """
    print('-' * n_comment_dash)
    print('Detect particles using the Circular Hough Transform (CHT)')
    print('-' * n_comment_dash)
    print(' ')

    im_copy = np.copy(im)
    im_blur = cv.medianBlur(im_copy, 3)
    # smooth to reduce noise a bit more
    # cv.Smooth(processed, processed, cv.CV_GAUSSIAN, 7, 7)
    # im_color = cv.cvtColor(im, cv.COLOR_GRAY2BGR)

    """
     The parameters for the opencv Circular Hough Transform are:
     cv.HoughCircles(
        image,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=10.0,
        param1=100,
        param2=30,
        minRadius=6,
        maxRadius=7)
    """
    min_radius = int(0.5*round(radius))
    max_radius = int(1.5*round(radius))
    circle_list = cv.HoughCircles(im_blur,
                                  cv.HOUGH_GRADIENT,
                                  dp=1.2,
                                  minDist=int(2*radius),
                                  param1=100,
                                  param2=30,
                                  minRadius=min_radius,
                                  maxRadius=max_radius)
    return circle_list, im_blur


def detection_laplace(im, radius):
    """

    :param im:
    :param radius:
    :return:
    """
    circle_list = []
    return circle_list


def detection(orig, method, saturation_perc, radius):
    """
    Detection of particles as centers and radii. Uses a specified method and does some pre-processing of the data.

    :param orig: The original image (can be RGB or gray-valued)
    :param method: On of the valid methods: ['CHT', 'Laplace']
    :param saturation_perc: Saturation percentage
    :param radius: Expected radius
    :return: [circle_list, im_gray, im_norm, im_blur]
        circle_list:
    """
    print('-' * n_comment_dash)
    print('Detect particles and return their centers and detected radius (if any) in a list')
    print('-' * n_comment_dash)
    print(' ')

    if orig.shape[-1] == 3:  # color image
        b, g, r = cv.split(orig)  # get b,g,r
        rgb_im = cv.merge([r, g, b])  # switch it to rgb
        im = cv.cvtColor(rgb_im, cv.COLOR_BGR2GRAY)
    else:
        im = np.copy(orig)

    im_gray = np.copy(im)

    # Contrast stretching
    perc_low, perc_high = np.percentile(im, (saturation_perc, 100-saturation_perc))
    im_norm = exposure.rescale_intensity(im, in_range=(perc_low, perc_high))
    # im_norm = cv.normalize(im, None, alpha=100, beta=200, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    # ax[1].imshow(im_norm, cmap=plt.cm.gray)

    if method == "CHT":
        circle_list, im_blur = detection_cht(im_norm, radius)
    elif method == "Laplace":
        circle_list, im_blur = detection_laplace(im_norm, radius)
    else:
        circle_list = []
        im_blur = np.copy(im)

    return circle_list, im_gray, im_norm, im_blur
