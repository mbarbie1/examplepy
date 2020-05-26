"""
-------------------------------------------------------------------------
Example script for feature extraction and comparison of features
-------------------------------------------------------------------------
Here we show the results of different features calculated from the particle
positions or the images itself. This is to give an initial idea of what the
different features mean to us and how we can use them.

.. moduleauthor:: Michael Barbier <michael@unam.bilkent.edu.tr>
"""

# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay
from time import time
import argparse

# Personal modules
from mlpy import util, detection, mio, plots, patterns as pr

# Reload personal modules (see: https://docs.python.org/3.8/tutorial/modules.html)
import importlib
importlib.reload(util)
importlib.reload(mio)
importlib.reload(plots)
importlib.reload(detection)
importlib.reload(pr)

# Print the OpenCV version for debugging
print(cv.__version__)


# -------------------------------------------------------------------------

"""
# -------------------------------------------------------------------------
# Script
# -------------------------------------------------------------------------
"""


util.my_comment('Load image with crystal (hexagonal) and gaseous state')
data_folder = '../data/phase_states'
mask_folder = '../data/phase_states/mask'
image_name = 'quasi_1.png'
#image_name = 'hexa_gaseous.png'
im = cv.imread(os.path.join(data_folder, image_name), 0)
mask_gt = cv.imread(os.path.join(mask_folder, image_name), 0)

util.my_comment('Detection of the particles in the image')
saturation_perc = 0.1
radius = 2
is_dark = 0
circle_list, im_gray, im_norm, im_blur = detection.detection(im, "Laplace", saturation_perc, radius, is_dark)
im_circles = np.copy(im_gray)
radius_show = 4
plots.draw_circles(circle_list, im_circles, radius=radius_show)
# Plot the detected particles
plt.subplot(121), plt.imshow(im, cmap='gray')
plt.title('Original image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(im_circles)
plt.title('Detection'), plt.xticks([]), plt.yticks([])
#plt.show()


util.my_comment('Calculation of features')

util.my_sub_comment('Lindemann features')
lindemann_list_5 = pr.compute_lindemann_parameter(circle_list, 5*radius)
lindemann_list_10 = pr.compute_lindemann_parameter(circle_list, 10*radius)
lindemann_list_20 = pr.compute_lindemann_parameter(circle_list, 20*radius)

util.my_sub_comment('Voronoi diagram features')
circle_list_array = np.array(circle_list)
centers = circle_list_array[:, 0:2]
n_centers = centers.shape[0]
vor = pr.compute_voronoi_diagram(centers)

#print(vor.point_region)

max_area = 1000.0
min_length_ratio = 0.01
[point_list, vertices_list, inside_point_list, areas_list, n_neighbors_list] = pr.features_voronoi(im, vor, n_centers, max_area)
norm_vv_list, n_neighbors_norm_list = pr.normalized_sides_voronoi(vertices_list, min_length_ratio)

"""
ax1 = plt.subplot(111)
plt.imshow(im, cmap='gray')
plots.plot_polygons(vertices_list, areas_list, ax=ax1, alpha=0.2, linewidth=0.7)

ax2 = plt.subplot(111)
plt.imshow(im, cmap='gray')
plots.plot_polygons(vertices_list, n_neighbors_list, ax=ax2, alpha=0.2, linewidth=0.7)

ax3 = plt.subplot(111)
plt.imshow(im, cmap='gray')
plots.plot_polygons(vertices_list, inside_point_list, ax=ax3, alpha=0.2, linewidth=0.7)

ax4 = plt.subplot(111)
plt.imshow(im, cmap='gray')
plots.plot_polygons(vertices_list, n_neighbors_norm_list, ax=ax4, alpha=0.2, linewidth=0.7)
"""

#fig = voronoi_plot_2d(vor, show_points=True, show_vertices=True, s=4)
#plt.show()


util.my_sub_comment('Delaunay triangulation features')
neighbors_list, dd_list, dd_mean, dd_std, aa_list, aa_mean, aa_std = pr.features_delaunay(centers)

df = pd.DataFrame(
   {
      'index': pd.Series(range(len(centers))),
      'center_x': pd.Series(centers[:, 0], dtype='float32'),
      'center_y': pd.Series(centers[:, 1], dtype='float32'),
      'valid_point': pd.Series(inside_point_list),
      'area': pd.Series(areas_list, dtype='float32'),
      'n_norm_NN': pd.Series(n_neighbors_norm_list),
      'n_NN': pd.Series(n_neighbors_list),
      'NN_distance_mean': pd.Series(dd_mean, dtype='float32'),
      'NN_distance_std': pd.Series(dd_std, dtype='float32'),
      'NN_angle_mean': pd.Series(aa_mean, dtype='float32'),
      'NN_angle_std': pd.Series(aa_std, dtype='float32'),
      'Lindemann_5': pd.Series(lindemann_list_5, dtype='float32'),
      'Lindemann_10': pd.Series(lindemann_list_10, dtype='float32'),
      'Lindemann_20': pd.Series(lindemann_list_20, dtype='float32'),
   })


#def intensity_points(im, centers):

centers_int = np.array(centers, np.int32)
intensity_list = mask_gt[centers_int[:, 1], centers_int[:, 0]]


#   return intensity_list

#plt.figure()
#df.plot()
#plt.legend(loc='best')
#plt.show()

a = 34
b = 45
"""
fig2 = plt.imshow(im, cmap='gray')
k = 100
#plt.triplot(centers[:, 0], centers[:, 1], tri.simplices)
plt.plot(centers[:, 0], centers[:, 1], 'o')
plt.plot(centers[k, 0], centers[k, 1], '*')
for nn in neighbors_list[k]:
    plt.plot(centers[nn, 0], centers[nn, 1], '*')

plt.show()
"""
