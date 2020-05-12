"""
-------------------------------------------------------------------------
Module lattice
-------------------------------------------------------------------------
Generate various crystal lattice structures.

.. module:: lattice
    :platform:
    :synopsis: Module for the generation of various crystal lattice structures.

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


def image_lattice(rr, shape, dot_radius):
    """
     Generates an image from given lattice points rr and lattice image size: shape, and radii of the dots: dot_radius

     :param rr: The lattice points as a list of (x, y) tuples
     :param shape: The corresponding lattice boundaries
     :param dot_radius: the radius of the plotted dots
     :return: The image of size shape with plotted lattice dots
    """
    img = np.zeros(shape, np.dtype('float'))
    n_rr = len(rr)
    for ii in range(0, n_rr):
        if dot_radius == 0:
            img[round(rr[ii][0]), round(rr[ii][1])] = 1.0
        else:
            rx = int(round(rr[ii][0]))
            ry = int(round(rr[ii][1]))
            cv.circle(img, (ry, rx), radius=dot_radius, color=1, thickness=cv.FILLED)
            #img = cv.circle(img, center, radius, color[, thickness[, lineType[, shift]]]    )

    return img


def generate_lattice(a1, a2, angle, n1, n2, shift_x, shift_y):
    """
    Generates a lattice of coordinates on a period lattice with lattice parameters a1, a2, under an angle angle

    :param a1: The lattice length parameter of the first basis vector (parallel with the x-axis)
    :param a2: The lattice length parameter second basis vector (under angle with the x-axis)
    :param n1: Number of repeating the basic lattice in x direction
    :param n2: Number of repeating the basic lattice in y direction
    :param angle: The angle between the 2 basis vectors of the unit lattice
    :param shift_x: Shift the dot in x direction (should be non-negative < a)
    :param shift_y: Shift the dot in y direction (should be non-negative < a)
    :return: The generated lattice
    """
    shape = (a1*n1, round(a2*math.sin(angle)*n2))
    rr = []
    #np.zeros((n1*n2, 2), np.dtype('float'))
    ii = 0
    for ix in range(-1, n1+1):
        for iy in range(-1, n2+1):
            rx = ix * a1 + iy * a2 * math.cos(angle) + shift_x
            ry = iy * a2 * math.sin(angle) + shift_y
            if rx < 0:
                rx = shape[0]+rx
            if rx > shape[0]:
                rx = rx-shape[0]
            if ry < 0:
                ry = shape[1]+ry
            if ry > shape[1]:
                ry = ry-shape[1]
            rr.append((rx, ry))
            ii = ii + 1
    return rr, shape
