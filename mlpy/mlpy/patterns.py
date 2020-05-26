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
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
import matplotlib.pyplot as plt
import pandas as pd

# Personal modules
from mlpy import util, detection

# Reload personal modules (see: https://docs.python.org/3.8/tutorial/modules.html)
import importlib
importlib.reload(util)
importlib.reload(detection)

n_comment_dash = 80
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------


def compute_lindemann_parameter(circle_list, radius):
    """
    Obtains the Lindemann parameter for a list of points (within a local radius).

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


def compute_voronoi_diagram(center_list):

    # Calculate Voronoi Polygons
    pp = np.array(center_list)
    vor = Voronoi(pp)

    return vor


def normalized_sides_voronoi(vertices_list, min_length):
    dv_list = [np.sqrt(np.square(vv[:, 0] - np.roll(vv[:, 0], 1)) + np.square(vv[:, 1] - np.roll(vv[:, 1], 1))) for vv in vertices_list]
    #dv_list = [np.sqrt(np.square(vv[:, 0]) + np.square(vv[:, 1])) for vv in vertices_list]
    norm_vv_list = [dv / np.sum(dv) for dv in dv_list]
    n_neighbors_norm_list = [np.sum(vv > min_length) for vv in norm_vv_list]

    return norm_vv_list, n_neighbors_norm_list


def features_voronoi(im, vor, n_centers, max_area):
    x_min = 0
    x_max = im.shape[0]
    y_min = 0
    y_max = im.shape[1]

    n_regions = len(vor.point_region)
    areas = np.zeros(shape=(n_centers+1))
    n_neighbors = np.zeros(shape=(n_centers+1))
    inside_point = np.zeros(shape=(n_centers+1))
    vertices = vor.vertices

    point_list = []
    vertices_list = []
    areas_list = []
    n_neighbors_list = []
    inside_point_list = []
    for jj in vor.point_region:
        vertex_indices_temp = np.array(vor.regions[jj])
        vertex_indices = np.delete(vertex_indices_temp, np.where(vertex_indices_temp == -1))
        vv = vertices[vertex_indices, :]

        inside_point[jj] = 1
        if np.min(vertex_indices_temp) != np.min(vertex_indices):
            inside_point[jj] = 0
        if np.min(vv[:, 1]) < 0 or np.max(vv[:, 1]) > x_max:
            inside_point[jj] = 0
        if np.min(vv[:, 0]) < 0 or np.max(vv[:, 0]) > y_max:
            inside_point[jj] = 0

        n_neighbors[jj] = len(vv)
        if (np.min(vertex_indices) > -1) and (len(vv) > 1) and (inside_point[jj] > 0):
            [areas[jj], orientation] = util.polyarea_signed(vv)
        else:
            areas[jj] = 0
            orientation = 'NA'
        vertices_list.append(vv)
        inside_point_list.append(inside_point[jj])
        areas_list.append(min(abs(areas[jj]), max_area))
        n_neighbors_list.append(n_neighbors[jj])
        point_list.append(jj)

    return point_list, vertices_list, inside_point_list, areas_list, n_neighbors_list


def features_delaunay(centers):
    tri = Delaunay(centers)
    (indptr, indices) = tri.vertex_neighbor_vertices
    
    neighbors_list = [indices[indptr[k]:indptr[k+1]] for k in range(0, len(centers))]
    rr_list = [centers[neighbors_list[k], :] - centers[k, :] for k in range(0, len(centers))]
    dd_list = [np.sqrt(np.square(rr[:, 0]) + np.square(rr[:, 1])) for rr in rr_list]
    aa_list = [np.arctan2(np.square(rr[:, 1]), np.square(rr[:, 0])) for rr in rr_list]
    dd_mean = [np.mean(dd) for dd in dd_list]
    dd_std = [np.std(dd) for dd in dd_list]
    aa_mean = [np.mean(np.abs(aa - np.roll(aa, 1, 0))) for aa in aa_list]
    aa_std = [np.std(np.abs(aa - np.roll(aa, 1, 0))) for aa in aa_list]
    
    return neighbors_list, dd_list, dd_mean, dd_std, aa_list, aa_mean, aa_std


def extract_features(im, mask):
    util.my_sub_comment('Detection of the particles in the image')
    saturation_perc = 0.1
    radius = 2
    is_dark = 0
    circle_list, im_gray, im_norm, im_blur = detection.detection(im, "Laplace", saturation_perc, radius, is_dark)

    #util.my_sub_comment('Calculation of features')

    util.my_sub_comment('Lindemann features')
    lindemann_list_5 = compute_lindemann_parameter(circle_list, 5 * radius)
    lindemann_list_10 = compute_lindemann_parameter(circle_list, 10 * radius)
    lindemann_list_20 = compute_lindemann_parameter(circle_list, 20 * radius)

    util.my_sub_comment('Voronoi diagram features')
    circle_list_array = np.array(circle_list)
    centers = circle_list_array[:, 0:2]
    n_centers = centers.shape[0]
    vor = compute_voronoi_diagram(centers)
    max_area = 1000.0
    min_length_ratio = 0.01
    [point_list, vertices_list, inside_point_list, areas_list, n_neighbors_list] = features_voronoi(im, vor,
                                                                                                       n_centers,
                                                                                                       max_area)
    norm_vv_list, n_neighbors_norm_list = normalized_sides_voronoi(vertices_list, min_length_ratio)

    util.my_sub_comment('Delaunay triangulation features')
    neighbors_list, dd_list, dd_mean, dd_std, aa_list, aa_mean, aa_std = features_delaunay(centers)
    nn_dd_mean = [np.mean(np.array(dd_mean)[np.array(neighbors, np.int32)]) for neighbors in neighbors_list]
    nn_dd_std = [np.mean(np.array(dd_std)[np.array(neighbors, np.int32)]) for neighbors in neighbors_list]
    nn_aa_mean = [np.mean(np.array(aa_mean)[np.array(neighbors, np.int32)]) for neighbors in neighbors_list]
    nn_aa_std = [np.mean(np.array(aa_std)[np.array(neighbors, np.int32)]) for neighbors in neighbors_list]

    df = pd.DataFrame(
        {
            'index': pd.Series(range(len(centers))),
            'center_x': pd.Series(centers[:, 0], dtype='float32'),
            'center_y': pd.Series(centers[:, 1], dtype='float32'),
            'valid_point': pd.Series(inside_point_list),
            'area': pd.Series(areas_list, dtype='float32'),
            'n_norm_NN': pd.Series(n_neighbors_norm_list),
            'n_NN': pd.Series(n_neighbors_list),
            'distance_mean': pd.Series(dd_mean, dtype='float32'),
            'distance_std': pd.Series(dd_std, dtype='float32'),
            'angle_mean': pd.Series(aa_mean, dtype='float32'),
            'angle_std': pd.Series(aa_std, dtype='float32'),
            'NN_distance_mean': pd.Series(nn_dd_mean, dtype='float32'),
            'NN_distance_std': pd.Series(nn_dd_std, dtype='float32'),
            'NN_angle_mean': pd.Series(nn_aa_mean, dtype='float32'),
            'NN_angle_std': pd.Series(nn_aa_std, dtype='float32'),
            'Lindemann_5': pd.Series(lindemann_list_5, dtype='float32'),
            'Lindemann_10': pd.Series(lindemann_list_10, dtype='float32'),
            'Lindemann_20': pd.Series(lindemann_list_20, dtype='float32'),
        })

    centers_int = np.array(centers, np.int32)
    y = mask[centers_int[:, 1], centers_int[:, 0]]

    return df, y

