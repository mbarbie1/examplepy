"""
-------------------------------------------------------------------------
Example script for the crystal pattern recognition (usage of importing personal modules)
-------------------------------------------------------------------------
This is an example script for:
  - reading and showing video data in Python.
  - Detecting particles
  - Showing circles on an image
  - Lindemann parameter calculation TODO
This version shows the usage of other (personal) modules
https://docs.python.org/3.8/tutorial/modules.html

.. moduleauthor:: Michael Barbier <michael@unam.bilkent.edu.tr>
"""

# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
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


def process(data_folder, file_path, frame_list, options):
    """
    This function processes the data:
        - Loads the selected frames of the video data,
        - Detects the particles within these image frames,
        - Shows the detected particles
        - TODO: the Lindemann formula we use seems not totally correct: if there is only one other particle, L equals to 0

    :param data_folder: The location of the input folder (of the data)
    :param file_path: The File path of the video data
    :param frame_list: A selection of frame indices (zero-based) from the video
    :param options: The options given to the detection, Lindemann, etc methods
    :return:
    """

    # Getting and the meta data from the file
    mio.print_metadata(file_path)
    meta = mio.get_metadata(file_path)

    # Reading the frames of the video
    frames = mio.read_frames(file_path, frame_list)

    # Single frame as example
    orig = frames[0]

    # We will do the particle detection in the frame for different parameters of contrast stretching the original image.
    #   n_param is the number of parameters (saturation percentage) tried for the contrast stretching
    n_param = 3
    saturation_perc_list = np.linspace(0.0, 2.0, num=n_param).tolist()
    for i_param in range(0, n_param):
        saturation_perc = saturation_perc_list[i_param]
        im = np.copy(orig)
        circle_list, im_gray, im_norm, im_blur = detection.detection(im, 'CHT', saturation_perc, radius=options.radius_detection)
        im_circles = np.copy(orig)
        plots.draw_circles(circle_list, im_circles, radius=options.radius_show)
        lindemann_list = pr.compute_lindemann_parameter(circle_list, 20*options.radius)
        #print(lindemann_list)
        histogram = plots.plot_lindemann_histogram(circle_list, options.radius)
        if options.show_method == 'Matplotlib':
            # Matplotlib will show four images: the grey-valued image of the original, the one after contrast stretching,
            #   the smoothed out version of the latter, and the original image with the resulting particle positions as
            #   circles.
            fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(16, 6))
            ax[0].imshow(im_gray, cmap=plt.cm.gray)
            ax[1].imshow(im_norm, cmap=plt.cm.gray)
            ax[2].imshow(im_blur, cmap=plt.cm.gray)
            ax[3].imshow(im_circles, cmap=plt.cm.gray)
            plt.show(block=False)
        elif options.show_method == 'OpenCV':
            # Show the annotated image using OpenCV instead, only the last image is shown. Press a key to continue
            cv.imshow("original with circles: saturation parameter = " + str(saturation_perc), im_circles )
            cv.waitKey(0)

    # Block until plots are closed
    plt.show()

def get_parameters():
    """
    Defines the (default) parameters used

    :return: The parameters in the options argparse class
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", help="training processing data path", default="/Users/mbarbier/Documents/data/test_data_camera/2020_03_02_fluorescent")
    parser.add_argument("--output_folder", help="output directory", default="./output")
    parser.add_argument("--file_name", help="File name of the video", default="JPEGcomp.avi")
    parser.add_argument("--video_index_first", help="First frame index", default=0, type=int)
    parser.add_argument("--video_index_last", help="Last frame index", default=5, type=int)
    parser.add_argument("--show_method_index", help="Method to use to show the detected particles [0:'Matplotlib', 1:'OpenCV']", default=0, type=int)
    parser.add_argument("--radius_detection", help="Radius used for the detection method, can be slightly different from the real radius and depends on the detection method", default=12, type=int)
    parser.add_argument("--radius_show", help="Radius shown on the annotated images", default=8, type=int)
    parser.add_argument("--radius", help="Real radius of the particles (in pixels)", default=6.25, type=float)
    options = parser.parse_args()
    options.frame_list = range(options.video_index_first, options.video_index_last+1)
    options.file_path = os.path.join(options.data_folder, options.file_name)
    options.show_method_list = ['Matplotlib', 'OpenCV']
    options.show_method = options.show_method_list[options.show_method_index]

    return options


def main():
    """
    Main function loading the parameters and processing the data

    :return:
    """
    options = get_parameters()
    process(options.data_folder, options.file_path, options.frame_list, options)


if __name__ == '__main__':
    main()

