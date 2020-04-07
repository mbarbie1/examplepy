"""
-------------------------------------------------------------------------
Module plots
-------------------------------------------------------------------------
Module for plots, graphs, annotation of images

.. module:: plots
    :platform:
    :synopsis: Module for plots, graphs, annotation of images

.. moduleauthor:: Michael Barbier <michael@unam.bilkent.edu.tr>
"""

# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import numpy as np
import cv2 as cv

print(cv.__version__)
n_comment_dash = 80
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------

def draw_circles(circle_list, im, radius):
    """
    Draw circles on an image with a specific radius.

    :param circle_list: List of [x,y,radius] of the circle
    :param im: The input image
    :param radius: The radius of the circles
    :return: The image with the circles annotated (writes on the original image)
    :rtype: image as uint8 rgb numpy matrix
    """
    print('-' * n_comment_dash)
    print('Annotate an image with circles')
    print('-' * n_comment_dash)
    print(' ')
    if circle_list is not None:
        circles = np.uint16(np.around(circle_list))
        for circle in circles:
            center = (circle[0], circle[1])
            # circle center
            #cv.circle(im, center, 1, (0, 0, 255), 1)
            # circle outline
            #radius = circle[2]
            cv.circle(im, center, radius, (255, 0, 255), 1)

    return im

def plot_lindemann_histogram(lindemann_parameter_list, n_bins):
    """ Plots the Lindemann histogram TODO """
    histogram = []
    return histogram