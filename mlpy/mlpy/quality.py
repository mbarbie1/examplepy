"""
-------------------------------------------------------------------------
Module quality
-------------------------------------------------------------------------
Verify various quality features of the videos: check for trembling of the camera, bubble locations, drift, spatial
statistics to check randomness of positions.

.. module:: quality
    :platform:
    :synopsis: Module for the verification of various quality characteristics of the videos: check for trembling of the
        camera, bubble locations, drift, spatial statistics to check randomness of positions.

.. moduleauthor:: Michael Barbier <michael@unam.bilkent.edu.tr>
"""


# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import io, filters


# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------
def get_bubble_mask(im_list, min_size):
    """
    Computes the standard deviation of a series of images of the video. The parts of the video which are
    not moving in time (no particles) could be a bubble or particles which are stuck. We are especially interested in
    bubble regions or dried particle regions which should be avoided or ignored by the detection.

    :param im_list: A list of frames
    :return: A mask which shows the larger (larger than min_size) zero movement regions (bubbles?)
    """
    mask = 0.0
    return mask


def trembling_velocity(im_list):
    """
    The displacement of each frame compared to the previous, this is done by detecting particles, and averaging their
    velocity at each time point.

    :param im_list: A consecutive list of frames
    :return: The average velocity of all the particles at a certain time point. Returns the list of them in x and
        y direction.
    """
    vx_list = [0.0]
    vy_list = [0.0]
    return vx_list, vy_list


def bg_illumination(im_list):
    bg_image = 0.0
    return bg_image


def position_randomness(im):
    randomness = 1.0
    return randomness
