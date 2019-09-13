#!/usr/bin/env python3

from setuptools import setup
import sys

__author__ = "Cesar Herrera"
__copyright__ = "Copyright (C) 2019 Cesar Herrera"
__license__ = "GNU GPL"

VERSION = "0.8"
DESCRIPTION = """Crabspy is a python library that provides a programming interface for tracking, counting and measuring 
crabs and other small creatures in videos recorded on natural and artificial settings. 
Crabspy is distributed under a GNU GPL license."""

PYTHON_VERSION = sys.version_info[:2]
PYTHON_REQUIRED = (3, 5)

if PYTHON_VERSION < PYTHON_REQUIRED:
    sys.stderr.write("""
    ==========================
    Your Python version is not supported.
    ==========================
    
    Crabspy requires Python {}.{}, and your current installed Python is {}.{}
""".format(*(PYTHON_REQUIRED + PYTHON_VERSION)))
    sys.exit(1)

with open("README.md") as file:
    LONG_DESCRIPTION = file.read()

setup(
    name="crabspy",
    version=VERSION,
    author=__author__,
    author_email="herrera.ce@gmail.com",
    url="https://github.com/CexyNature/Crabspy",
    download_url="https://github.com/CexyNature/Crabspy",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license=__license__,
    packages=["crabspy"],
    python_requires=">=3.5",
    install_requires=[
        "numpy>=1.13",
        "opencv-python>=3.3",
        "opencv-contrib-python>=3.3",
        "pandas>=0.21.0",
        "scikit-image >= 0.13.1",
        "scikit-learn >= 0.21.3"])
