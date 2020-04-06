# -------------------------------------------------------------------------
# EXAMPLE_OPEN_IMAGE
# -------------------------------------------------------------------------
# This is an example script for reading and showing video data in Python.
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
import cv2
import sys
import os
print(cv2.__version__)
from skimage import io, filters, measure, color
from skvideo import io
import ffmpeg
import json
n_comment_dash = 80
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------
data_folder = '/Users/mbarbier/Documents/data/test_data_camera/2020_03_02_fluorescent'
file_names = ['JPEGcomp.avi', 'DVcomp-90fps.mov', 'DVcomp-100fps.avi', 'from_nd2_to_avi_dv_comp.avi', 'from_nd2_to_avi_no_comp.avi', 'from_nd2_to_avi_jpeg_comp.avi']
file_path = os.path.join(data_folder, file_names[0])
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# Example scripts
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
print('-'*n_comment_dash)
print('Print metadata using ffmpeg tool')
print('-'*n_comment_dash)
print(' ')

meta = ffmpeg.probe(file_path)
print(meta)
print(json.dumps(meta, indent=4))

print('-'*n_comment_dash)
print(' ')
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
print('-'*n_comment_dash)
print('Load and show data/metadata using the OpenCV library')
print('-'*n_comment_dash)
print(' ')

try:
    f = cv2.VideoCapture(file_path)
except:
    print("problem opening input stream")
    sys.exit(1)
if not f.isOpened():
    print("capture stream not open")
    sys.exit(1)

frame_id = 0
n_frames = 50
while f.isOpened():

    frame_id = frame_id + 1
    ret, frame = f.read()

    if frame_id > n_frames:
        break

    if ret:
        cv2.imshow('Frame: ' + str(frame_id), frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

f.release()

print('-'*n_comment_dash)
print(' ')
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
print('-'*n_comment_dash)
print('Load and show data/metadata using the Scikit-Video library')
print(' ')
print('      Doesnt work on the Mac (codec problem)')
print('-'*n_comment_dash)

try:
    f = io.VideoCapture(file_path)
except:
    print("problem opening input stream")
    ret, frame = f.read()
    io.imshow('Frame loaded by skikit-video', frame)
    #sys.exit(1)
if not f.isOpened():
    print("capture stream not open")
    #sys.exit(1)

print('-'*n_comment_dash)
print(' ')
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
print('-'*n_comment_dash)
print('Annotate an image with circles')
print('-'*n_comment_dash)
print(' ')

# Load a single frame
try:
    f = cv2.VideoCapture(file_path)
except:
    print("problem opening input stream")
    sys.exit(1)
if not f.isOpened():
    print("capture stream not open")
    sys.exit(1)

frame_id = 0
n_frames = 50
while f.isOpened():

    frame_id = frame_id + 1
    ret, frame = f.read()

im = cv2.circle(im, (100, 400), 20, (255, 0, 0), 3)

print('-'*n_comment_dash)
print(' ')
# -------------------------------------------------------------------------
