"""
-------------------------------------------------------------------------
Example script for feature extraction and supervised machine learning
-------------------------------------------------------------------------
Here we apply machine learning to some example images and test the
results of test samples.

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
from sklearn import manifold, datasets, svm
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

data_folder = '../data/phase_states'
mask_folder = '../data/phase_states/mask'
image_name_list = ['quasi_1.png', 'square_1.png', 'rect_2.png', 'hexa_1.png', 'hexa_2.png']
feature_folder = '../data/phase_states/features'
feature_file_name = 'features_1.csv'
do_read_feature_table = True

if do_read_feature_table:
    util.my_comment('Load training features from table')
    df_train = pd.read_csv(os.path.join(feature_folder, feature_file_name))
else:
    util.my_comment('Load training images and save feature table')
    df_list = []
    for image_name in image_name_list:
        im = cv.imread(os.path.join(data_folder, image_name), 0)
        mask_gt = cv.imread(os.path.join(mask_folder, image_name), 0)
        df, y = pr.extract_features(im, mask_gt)
        df['image_name'] = image_name
        df['gt'] = y
        df_list.append(df)

    df_train = pd.concat(df_list)
    df_train.to_csv(os.path.join(feature_folder, feature_file_name), index=False)

util.my_comment('Extract trainable features of the data')
feature_list = [
    'area',
    'n_norm_NN',
    'n_NN',
    'NN_distance_mean',
    'NN_distance_std',
    'NN_angle_mean',
    'NN_angle_std',
    'Lindemann_5',
    'Lindemann_10',
    'Lindemann_20'
]
X = df_train.loc[:, feature_list]
y = df_train.loc[:, 'gt']

util.my_comment('Train an SVM')


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

   Parameters
   ----------
   x: data to base x-axis meshgrid on
   y: data to base y-axis meshgrid on
   h: stepsize for meshgrid, optional

   Returns
   -------
   xx, yy : ndarray
   """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

   Parameters
   ----------
   ax: matplotlib axes object
   clf: a classifier
   xx: meshgrid ndarray
   yy: meshgrid ndarray
   params: dictionary of params to pass to contourf, optional
   """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
util.my_comment('Train an SVM')
clf = models[0]
model = clf.fit(X, y)
util.my_comment('Finished training')


"""
# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

XX = df_train[df_train["image_name"] == 'quasi_1.png']
X0, X1 = XX.iloc[:, 1], XX.iloc[:, 2]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
"""