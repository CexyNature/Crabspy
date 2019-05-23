#!/usr/bin/env python3

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

"""
List of constants used in other modules.
These should be define by the user to fit their experimental setting and purpose.

DIM:
    Store the quadrat's dimension (sides) in space units such as centimeters or meters.
    All sides must be equal (i.e. square).

COLOR_SET_0:
    Define the color to use to draw reference elements in the image, such as:
    digital quadrat, point vertices, mouse position, and organism position.

COLOR_SET_1:
    Alternative color to use at user discretion.

CAPTURE_VERTICES: bool
    If True will enable the capture of quadrat's vertices in the image using the mouse click function.
    If False will import quadrat's vertices coordinates provided by user (i.e. QUADRAT_POINTS)

QUADRAT_POINTS:
    Known pixel coordinates of quadrat's vertices on image.

MANUAL_ANNOTATION:
    If True user input crab_name, crab_species, sex and crab_handedness.
    If False a random name is given to crab_name and other properties are set to "unknown".
"""

DIM = [50, 50, 50, 50]
COLOR_SET_0 = (243, 28, 20)
COLOR_SET_1 = (18, 250, 173)
CAPTURE_VERTICES = False
QUADRAT_POINTS = [(628, 105), (946, 302), (264, 393), (559, 698)]
MANUAL_ANNOTATION = False
ERODE = (3, 3)
DILATE = (7, 5)
