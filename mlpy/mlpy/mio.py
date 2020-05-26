"""
-------------------------------------------------------------------------
Module mio
-------------------------------------------------------------------------
Module for input/output of images, text files, etc

.. module:: mio
    :platform:
    :synopsis: Module for input/output of images, text files, etc

.. moduleauthor:: Michael Barbier <michael@unam.bilkent.edu.tr>
"""

# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import cv2 as cv
import sys
import ffmpeg
import json

n_comment_dash = 80
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------

def get_metadata(file_path):
    """
    Extracts the meta data of an image/video using ffmpeg and puts it into a specific format (dictionary)

    :param file_path: File path of the image/video
    :return: The meta data as a dictionary
    """
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
    """
    Prints the meta data as extracted by ffmpeg and returns this raw meta data.

    :param file_path: File path of the image/video
    :return: The meta data as extracted by ffmpeg
    """
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
    """
    Loads video data using the OpenCV library (reads in a specified list of frames).

    :param file_path: File path of the video
    :param frame_list: List of frames (the indices) of interest
    :return: A list with the frames as numpy? data
    """
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
