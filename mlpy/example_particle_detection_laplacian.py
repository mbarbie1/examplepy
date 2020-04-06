# -------------------------------------------------------------------------
# EXAMPLE_OPEN_IMAGE
# -------------------------------------------------------------------------
# This is an example script for:
#   - reading and showing video data in Python.
#   - Detecting particles
#   - Showing circles on an image
#   - Lindemann parameter calculation TODO
#
# Author: Michael Barbier
#
# Version history:
#
#   version v1: Original version.
#
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import cv2 as cv
import sys
import os
import numpy as np

print(cv.__version__)
from skimage import io, filters, measure, color, exposure
from skvideo import io
import ffmpeg
import json
import matplotlib.pyplot as plt
from scipy.spatial import distance
import copy

n_comment_dash = 80
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# functions
# -------------------------------------------------------------------------

def get_metadata(file_path):
    print('-' * n_comment_dash)
    print('Get metadata using ffmpeg tool')
    print('-' * n_comment_dash)
    print(' ')

    tmp_meta = ffmpeg.probe(file_path)
    tmp = tmp_meta["streams"][0]
    print(tmp)
    meta = {}
    meta['pixel_size_x'] = -1
    meta['pixel_size_y'] = -1
    meta['pixel_size_z'] = -1
    meta['frame_rate'] = float(tmp['nb_frames']) / float(tmp['duration'])
    meta['time_interval'] = float(tmp['duration']) / float(tmp['nb_frames'])
    meta['size_x'] = tmp['width']
    meta['size_y'] = tmp['height']
    meta['size_c'] = -1
    meta['size_t'] = tmp['nb_frames']
    meta['size_z'] = -1
    meta['bitdepth'] = tmp['bits_per_raw_sample']

    print()
    print(json.dumps(meta, indent=4))

    print('-' * n_comment_dash)
    print(' ')
    return meta


def print_metadata(file_path):
    print('-' * n_comment_dash)
    print('Print metadata using ffmpeg tool')
    print('-' * n_comment_dash)
    print(' ')

    meta = ffmpeg.probe(file_path)
    print(json.dumps(meta, indent=4))

    print('-' * n_comment_dash)
    print(' ')
    return meta


def read_frames(file_path, frame_list):
    print('-' * n_comment_dash)
    print('Load data using the OpenCV library')
    print('-' * n_comment_dash)
    print(' ')

    try:
        f = cv.VideoCapture(file_path)
    except:
        print("problem opening input stream")
        sys.exit(1)
    if not f.isOpened():
        print("capture stream not open")
        sys.exit(1)

    frame_id = 0
    frames = []
    n_frames = len(frame_list)
    for frame_id in frame_list:

        if f.isOpened():
            f.set(cv.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = f.read()
            if ret:
                frames.append(frame)
            else:
                break

    f.release()

    print('-' * n_comment_dash)
    print(' ')
    return frames


def detection_cht(im, radius):
    print('-' * n_comment_dash)
    print('Detect particles using the Circular Hough Transform (CHT)')
    print('-' * n_comment_dash)
    print(' ')

    im_copy = np.copy(im)
    im_blur = cv.medianBlur(im_copy, 3)
    # smooth to reduce noise a bit more
    # cv.Smooth(processed, processed, cv.CV_GAUSSIAN, 7, 7)
    #im_color = cv.cvtColor(im, cv.COLOR_GRAY2BGR)

    """
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
    return circle_list, im_blur


def detection_laplace(im, radius):
    circle_list = []
    return circle_list


def detection(orig, method, saturation_perc, radius):
    """ Detection  """
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

    im_gray = np.copy(im)

    # Contrast stretching
    perc_low, perc_high = np.percentile(im, (saturation_perc, 100-saturation_perc))
    im_norm = exposure.rescale_intensity(im, in_range=(perc_low, perc_high))
    #im_norm = cv.normalize(im, None, alpha=100, beta=200, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    #ax[1].imshow(im_norm, cmap=plt.cm.gray)

    if method == "CHT":
        circle_list, im_blur = detection_cht(im_norm, radius)
    elif method == "Laplace":
        circle_list, im_blur = detection_laplace(im_norm, radius)
    else:
        circle_list = []
        im_blur = np.copy(im)

    return circle_list, im_gray, im_norm, im_blur


def draw_circles(circle_list, im, radius):
    """ Draw circles on an image with a specific radius. """
    print('-' * n_comment_dash)
    print('Annotate an image with circles')
    print('-' * n_comment_dash)
    print(' ')
    if circle_list is not None:
        circles = np.uint16(np.around(circle_list))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            # circle center
            #cv.circle(im, center, 1, (0, 0, 255), 1)
            # circle outline
            #radius = circle[2]
            cv.circle(im, center, radius, (255, 0, 255), 1)

    return im


def compute_lindemann_parameter(circle_list, radius):
    """ Compute the Lindemann parameter for each circle, TODO """
    circles = np.asarray(circle_list[0])
    points = circles[:, 0:2]
    dist = distance.cdist(points, points, 'euclidean')
    n_points = dist.shape[0]
    dist_mask = np.ma.masked_outside(dist, 0.01, radius)
    L = np.zeros((n_points))
    for j in range(0, n_points):
        d = dist_mask[j, :]
        print(d)
        N = (d > 0.01).sum()
        print(N)
        if N > 0:
            d2 = np.square(d)
            mean_d = d.mean()
            mean_d2 = d2.mean()
            L[j] = np.sqrt(np.abs(mean_d2 - np.square(mean_d))) / mean_d
        else:
            L[j] = 1
    lindemann_list = L.tolist()

    return lindemann_list

def plot_lindemann_histogram(lindemann_parameter_list, n_bins):
    """ Plots the Lindemann histogram TODO """
    histogram = []
    return histogram

# -------------------------------------------------------------------------
# Example scripts
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------
data_folder = '/Users/mbarbier/Documents/data/test_data_camera/2020_03_02_fluorescent'
file_names = ['JPEGcomp.avi', 'DVcomp-90fps.mov', 'DVcomp-100fps.avi', 'from_nd2_to_avi_dv_comp.avi',
              'from_nd2_to_avi_no_comp.avi', 'from_nd2_to_avi_jpeg_comp.avi']
file_path = os.path.join(data_folder, file_names[0])

# -------------------------------------------------------------------------

# Getting and the meta data from the file
print_metadata(file_path)
meta = get_metadata(file_path)

# The list of frame to be loaded
frame_list = range(0, 5)
# Reading the frames of the video
frames = read_frames(file_path, frame_list)

# Single frame as example
orig = frames[0]
# The radius given to the detection algorithm (the radius is too large)
radius_detection = 12
# Chose a method for the plotting of the results
show_method_list = ['Matplotlib', 'OpenCV']
show_method = show_method_list[0]
# The radius used in the plots
radius_show = 8
# The real radius in pixels of the particles
radius = 6.25

# We will do the particle detection in the frame for different parameters of contrast stretching the original image.
#   n_param is the number of parameters (saturation percentage) tried for the contrast stretching
n_param = 3
saturation_perc_list = np.linspace(0.0, 2.0, num=n_param).tolist()
for i_param in range(0, n_param):
    saturation_perc = saturation_perc_list[i_param]
    im = np.copy(orig)
    circle_list, im_gray, im_norm, im_blur = detection(im, 'CHT', saturation_perc, radius=radius_detection)
    im_circles = np.copy(orig)
    draw_circles(circle_list, im_circles, radius=radius_show)
    lindemann_list = compute_lindemann_parameter(circle_list, 20*radius)
    print(lindemann_list)
    histogram = plot_lindemann_histogram(circle_list, radius)
    if show_method == 'Matplotlib':
        # Matplotlib will show four images: the grey-valued image of the original, the one after contrast stretching,
        #   the smoothed out version of the latter, and the original image with the resulting particle positions as
        #   circles.
        fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(16, 6))
        ax[0].imshow(im_gray, cmap=plt.cm.gray)
        ax[1].imshow(im_norm, cmap=plt.cm.gray)
        ax[2].imshow(im_blur, cmap=plt.cm.gray)
        ax[3].imshow(im_circles, cmap=plt.cm.gray)
        plt.show(block=False)
    elif show_method == 'OpenCV':
        # Show the annotated image using OpenCV instead, only the last image is shown. Press a key to continue
        cv.imshow("original with circles: saturation parameter = " + str(saturation_perc), im_circles )
        cv.waitKey(0)

# Block until plots are closed
plt.show()


# -------------------------------------------------------------------------
