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
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm
import matplotlib as mpl

n_comment_dash = 80
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------


class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)


def draw_circles(circle_list, im, radius):
    """
    Draw circles on an image with a specific radius.

    :param circle_list: List of [x,y,radius] of the circle
    :param im: The input image
    :param radius: The radius of the circles
    :return: The image with the circles annotated (writes on the original image)
    :rtype: image as uint8 rgb numpy matrix
    """
    #print('-' * n_comment_dash)
    #print('Annotate an image with circles')
    #print('-' * n_comment_dash)
    #print(' ')
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


def plot_polygons(polygons: object, values: object, ax: object = None, alpha: object = 0.5, linewidth: object = 0.7) -> object:
    min_value = np.min(values)
    max_value = np.max(values)

    COL = MplColorHelper('jet', min_value, max_value)

    if ax is None:
        plt.figure(figsize=(5, 5))
        ax = plt.subplot(111)
    ax.axis("equal")
    for jj in range(0, len(polygons)):
        poly = polygons[jj]
        colored_cell = Polygon(poly,
                               linewidth=linewidth,
                               alpha=alpha,
                               facecolor=COL.get_rgb(values[jj]),
                               edgecolor="black")
        ax.add_patch(colored_cell)

    plt.show()
