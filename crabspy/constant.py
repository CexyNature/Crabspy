#!/usr/bin/env python3

"""
Defines constants used by other modules.
"""

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
    
ERODE: tuple
    Kernel size for erosion morphological command.

DILATE: tuple
    Kernel size for dilation morphological command.

DECK: integer
    Size of double ended queue for smoothing track coordinate using a moving average.

SNAPSHOT: bool
    If True snapshots (one per frame) from tracked individual will be saved inside directory /results/snapshot/VIDEO-FILE-NAME/INDIVIDUAL-NAME/
    If False snapshots will not be saved.

RESIZE: numeric
    Resize value for applying to image.
    Value equal one (1) will conserve original image's dimension.
"""

DIM = [80, 80, 80, 80]
COLOR_SET_0 = (243, 28, 20)
COLOR_SET_1 = (18, 250, 173)
CAPTURE_VERTICES = False
# QUADRAT_POINTS = [(628, 105), (946, 302), (264, 393), (559, 698)]
QUADRAT_POINTS = [(461, 171), (910, 199), (348, 514), (853, 545)]
MANUAL_ANNOTATION = True
ERODE = (3, 3)
DILATE = (7, 5)
DECK = 10
SNAPSHOT = True
RESIZE = 1
