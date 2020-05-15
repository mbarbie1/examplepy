"""
-------------------------------------------------------------------------
Example script for the particle detection, feature extraction and t-SNE
-------------------------------------------------------------------------

.. moduleauthor:: Michael Barbier <michael@unam.bilkent.edu.tr>
"""

# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time
import argparse

# Personal modules
from mlpy import detection, mio, plots, patterns as pr

# Reload personal modules (see: https://docs.python.org/3.8/tutorial/modules.html)
import importlib
importlib.reload(mio)
importlib.reload(plots)
importlib.reload(detection)
importlib.reload(pr)

# Print the OpenCV version for debugging
print(cv.__version__)

# Constant number of dashes
n_comment_dash = 80
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Script
# -------------------------------------------------------------------------

# Load image with crystal (hexagonal) and gaseous state
im = cv.imread('../data/phase_states/hexa_gaseous.png', 0)
#im = cv.imread('../data/phase_states/quasi_gaseous.png', 0)

# Detection of the particles in the image
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
plt.show()

# Calculation of features
lindemann_list_5 = pr.compute_lindemann_parameter(circle_list, 5*radius)
lindemann_list_10 = pr.compute_lindemann_parameter(circle_list, 10*radius)
lindemann_list_20 = pr.compute_lindemann_parameter(circle_list, 20*radius)

# Calculation of the t-SNE
circles = np.array(circle_list)
color = np.array(lindemann_list_10)
features_1 = np.array(lindemann_list_5)
features_2 = np.array(lindemann_list_10)
features_3 = np.array(lindemann_list_10)
X = np.vstack((features_1, features_2, features_3))
X = np.transpose(X)

n_samples = 300
n_components = 2
(fig, subplots) = plt.subplots(1, 2, figsize=(15, 5))
perplexities = [50]#[5, 30, 50, 100]

ax = subplots[0]
ax.set_title("Crystal")
ax.imshow(im, cmap='gray')
ax.scatter(circles[:, 0], circles[:, 1], c=color, s=4)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())

for i, perplexity in enumerate(perplexities):
    ax = subplots[i + 1]

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    t1 = time()
    print("Original, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

    ax.set_title("Perplexity=%d" % perplexity)
    ax.scatter(Y[:, 0], Y[:, 1], c=color)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')

plt.show()


