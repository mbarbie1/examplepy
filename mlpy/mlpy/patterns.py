"""
-------------------------------------------------------------------------
Module patterns
-------------------------------------------------------------------------
The machine learning methods to recognize various crystal structures.

.. module:: patterns
    :platform:
    :synopsis: Module for machine learning techniques to recognize local crystal structure

.. moduleauthor:: Michael Barbier <michael@unam.bilkent.edu.tr>
"""

# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import numpy as np
import cv2 as cv
from scipy.spatial import distance

print(cv.__version__)
n_comment_dash = 80
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------


def compute_lindemann_parameter(circle_list, radius):
    """
    Draw circles on an image with a specific radius.

    :param circle_list: list of [x, y, radius], decribing the circles
    :param radius: The radius describing the size of the local region around the point/particle of interest
    :return: None
    """
    circles = np.asarray(circle_list)
    points = circles[:, 0:2]
    dist = distance.cdist(points, points, 'euclidean')
    n_points = dist.shape[0]
    dist_mask = np.ma.masked_outside(dist, 0.01, radius)
    L = np.zeros((n_points))
    for j in range(0, n_points):
        d = dist_mask[j, :]
        #print(d)
        N = (d > 0.01).sum()
        #print(N)
        if N > 0:
            d2 = np.square(d)
            mean_d = d.mean()
            mean_d2 = d2.mean()
            L[j] = np.sqrt(np.abs(mean_d2 - np.square(mean_d))) / mean_d
        else:
            L[j] = 1
    lindemann_list = L.tolist()

    return lindemann_list