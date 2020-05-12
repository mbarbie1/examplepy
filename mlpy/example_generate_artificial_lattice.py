"""
-------------------------------------------------------------------------
Example script: generating artificial crystal patterns
-------------------------------------------------------------------------
Artificial crystal lattice images can help us in understanding how different
patterns look, how their Fourier spectrum looks like and function as a ground
truth for the particle detection and pattern recognition by machine learning.

Here we generate simple artificial images: the simplest ones are white dots on
a dark background, more involved ones should resemble more real images.

.. moduleauthor:: Michael Barbier <michael@unam.bilkent.edu.tr>
"""

# -------------------------------------------------------------------------
# Import libraries
# -------------------------------------------------------------------------
import cv2 as cv
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Personal modules
from mlpy import lattice

# Reload personal modules (see: https://docs.python.org/3.8/tutorial/modules.html)
import importlib
importlib.reload(lattice)

def main():
    """
    Main function generating some example lattices

    :return:
    """
    PI = math.pi
    a1 = 64
    a2 = 64
    angle = 120.0 * PI / 180.0
    n1 = 8
    n2 = 6
    shift_x = a1 / 4
    shift_y = a2 * math.sin(angle) / 2.0
    dot_radius = 32
    rr1, shape = lattice.generate_lattice(a1, a2, angle, n1, n2, shift_x, shift_y)
    img1 = lattice.image_lattice(rr1, shape, dot_radius)
    rr2, shape = lattice.generate_lattice(a1, a2, angle, n1, n2, shift_x+round(a2/2.0), shift_y+round(a2/3.0))
    img2 = lattice.image_lattice(rr2, shape, dot_radius)

    plt.subplot(111), plt.imshow(img1+img2, cmap='gray')
    plt.title('Hexagonal'), plt.xticks([]), plt.yticks([])

    plt.show()

if __name__ == '__main__':
    main()
