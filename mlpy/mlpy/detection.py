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
from skimage import io, filters, feature, img_as_float, measure, color, exposure
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
    circle_list = circle_list[0, :]

    return circle_list, im_blur


def detection_laplace(im, radius):
    """
    Detect particles with a Laplace based spot detector
    (There are also blob detectors available in scikit-image but I was not successful in using them:
    DoG = Difference of Gaussian, LoG = Laplacian of Gaussian, DoH = Determinant of Hessian)
    TODO: we need an extra parameter for the threshold (or a very good automatic threshold)

    :param im: The input image which should be grey-valued
    :param radius: The radius of the particles used by the LoG (Laplacian of Gaussian)
    :return: [1,2]:
        (1) A list with [x, y, radius] values,
        (2) The smoothed image used as input to the Laplacian
    """
    circle_list = []
    im_blur = filters.gaussian(im, sigma=radius)
    im_lap = -cv.Laplacian(im_blur, cv.CV_64F)
    im = img_as_float(im_lap)
    centers = feature.peak_local_max(im, min_distance=radius, threshold_rel=0.05)
    circle_list = np.zeros((centers.shape[0], 3),)
    circle_list[:, 0] = centers[:, 1]
    circle_list[:, 1] = centers[:, 0]
    # circle_list = feature.blob_log(im_blur, max_sigma=(radius*3), num_sigma=radius, threshold=.1)
    # blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    circle_list[:, 2] = radius
    circle_list = circle_list.tolist()

    return circle_list, im_blur


def detection(orig, method, saturation_perc, radius, is_dark):
    """
    Detection of particles as centers and radii. Uses a specified method and does some pre-processing of the data.

    :param orig: The original image (can be RGB or gray-valued)
    :param method: On of the valid methods: ['CHT', 'Laplace']
    :param saturation_perc: Saturation percentage
    :param radius: Expected radius
    :param is_dark: Whether the appearence of the particles is dark or bright (1 or 0)
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

    if is_dark:
        im = np.invert(np.array(im, dtype=im.dtype))
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
