# Workflow and script utilities to generate 3D reconstructions

This document describes the workflow, python scripts and VisualSFM commands used to create photogrammetry reconstructions from sediment videos.

# Content:

- [Overview](#Overview)
- [Why assess sediment changes?](#Why-assess-sediment-changes?)
- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Installation](#Installation)
- [How to collaborate](#How-to-collaborate)
- [Acknowledgement](#Acknowledgement)

## Overview

This workflow consists of several steps: collecting videos, extracting frames from videos, selecting frames for reconstruction, and creating a sparse and dense reconstruction by feature detection and pairwise image matching. Descriptions in how to create a video for photogrammetry reconstructions can be found in the scientific literature and internet tutorials. Here, I have made available some python scripts which are useful for extracting and selecting video frames. All photogrammetry computations are done using VisualSFM, and a python script to iteratively run VisualSFM on several data sets is also available in this repository.

## Why assess sediment changes?

Invertebrates living in the benthos are among the most important turbation factors in marine ecosystems. Intertidal burrowing crabs can rework the sediment at incredible high rates. For instance, some crabs can rework the entire top 15 cm layer of sediment over the scale of days and/or months. In some ecosystems and settings this turnover rate seems to be a key factor in structuring the microbial community in the sediment and in driving plant and infauna composition growth rates. However, traditional ways to assess bioturbation require intricate and often complex sampling techniques. By assessing change in sediment volume using photogrammetry over short time intervals I seek to quantify the crab bioturbation activity. Thus, I propose change in sediment volume as a surrogate of turbation, in particular sediment turnover rate.

## Introduction



## Requirements

- python >= 3.5

- opencv-contrib-python==3.4.2.16

- VisualSFM

## Installation

Scripts in this repository required VisualSFM. Instructions on how to install and use VisualSFM can be found in its [webpage](http://ccwu.me/vsfm/index.html). Utilities in this submodule only required Python and OpenCV. Please observe that Crabspy requires additional packages.

Note to self: *Chronos* runs this module using a conda environment:

`conda activate py361cv330`

## How to collaborate



## Acknowledgement

This workflow would not been possible, or at least available as an open and free version, without the great contribution of [Changchang Wu](http://ccwu.me/) and his [VisualSFM](http://ccwu.me/vsfm/).
