[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3820270.svg)](https://doi.org/10.5281/zenodo.3820270)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub Fork: ](https://img.shields.io/github/forks/CexyNature/Crabspy?label=Fork&style=social)](https://github.com/CexyNature/Crabspy)
[![GitHub Start: ](https://img.shields.io/github/forks/CexyNature/Crabspy?label=Starts&style=social)](https://github.com/CexyNature/Crabspy)
[![Twitter](https://img.shields.io/twitter/follow/CexyNature?style=social)](https://twitter.com/cexynature?lang=en)

<img src="images/logos/pseudo_logo.jpg" width="150" height="150"> CrabSpy
==========

An heuristic toolbox for spying<sup>*</sup> intertidal crabs in their environment; built in Python. In continuous development.

<sup>*</sup> Furtively collect information about crabs functional biology and ecology: species identity, movement patterns, change in coloration, feeding rates, bioturbation, and more. 


> *"Until comparatively recently, ecologists were content to describe how nature “looks” (sometimes by means of fantastic
 terms!) and to speculate on what she might have looked like in the past or may look like in the future. 
 Now, an equal emphasis is being placed on what nature ‘does’, and rightly so, because the changing face 
 of nature can never be understood unless her metabolism is also studied. This change in approach brings 
 the small organisms into perspective with the large, and encourages the use of experimental methods to 
 supplement the analytic. It is evident that so long as a purely descriptive viewpoint is maintained, 
 there is very little in common between such structurally diverse organisms as sperma-tophytes, 
 vertebrates and bacteria. In real life, however, all these are intimately linked functionally in 
 ecological systems, according to well-defined laws. Thus the only kind of general ecology is that which
 I call a ‘functional ecology’..."*

> -- Eugene P. Odum <br>
> &nbsp;&nbsp;&nbsp; Fundamentals of Ecology, 1957

# Content:

- [Overview](#Overview)
- [Why CrabSpy and similar initiatives are important?](#Why-CrabSpy-and-similar-initiatives-are-important?)
- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Installation](#Installation)
- [How to collaborate](#How-to-collaborate)
- [Acknowledgement](#Acknowledgement)

## Overview

Crabspy is proposing an alternative workflow to collect biological and ecological data from intertidal crabs using computer vision and learning. Crabspy aims to provide a toolbox which can accelerate and improve information collection, enabling rapid and actionable scientific and policy response by means of a faster data streamline.


## Why CrabSpy and similar initiatives are important?

Traditional methods to collect data on the ecology of crabs and their role within ecosystems are often time consuming, invasive and can alter the natural habitat of the study site. The development of new technologies therefore presents an ideal opportunity to innovate and improve on traditional techniques to scale up data collection to the levels required. Furthermore, many species are disappearing before we can even catalogue them or describe them. Thus, natural scientists are faced with challenge of needing to rapidly increase their efforts to gather reliable species and ecosystem information at broader spatial and temporal scales. Crabspy seeks to ease some of these challenges by incorporating automation on data collection.

## Introduction

Crabspy was created to enable scientist to create their own image databases and train their own detection and classification models. Thus, most routines in Crabspy use simple off-the-shelf algorithms to extract data (e.g. OpenCV tracking methods). Manual methods are also included for those situations where simple algorithms are not successful.

Crabspy was originally developed to track intertidal crabs in soft sediments. However, other functions have been included such as counting feeding events, extracting individuals' colors, measuring and counting burrows, and more. Some of these routines require training machine learning algorithms.

Crabspy also includes some utilities for [organizing information associated to videos](/crabspy/toolkit/README.md), and a [workflow for extracting frames from videos and running 3D reconstructions using the popular VisualSFM program](/crabspy/toolkit/3Dmodels/README.md).

![](/images/examples/example_vid_output.gif)

## Requirements

Crabspy requires Python >= 3.5. Full list of dependencies can be found in the [requirements file](requirements.txt)

## Installation

1. Create virtual environment
2. Clone or Download project
3. Navigate inside project
4. Run

```
python setup.py install
```

## Work in Progress (yet not working as expected)

- correct_hc.py
- fast_track.py
- draw_track.py
- manual_tracking.py
- scoop_feed.py
- scoop_feed_v1.py
- snaps_dict.py
- svm_hog.py
- test.py



## How to collaborate

We would like to see this code used by other researchers, professionals and nature enthusiasts. Please do not hesitate to share your friendly feedback, create issues, suggest features, and create pull requests. We are also keen in receiving bugs reports and fiddler crab images.

*Information about submitting pull request can be found in this [article](https://code.tutsplus.com/tutorials/how-to-collaborate-on-github--net-34267).*


## Acknowledgement

Organizations which made possible this project by funding or allocating other sources (computing, expertise) to C. Herrera

<a href="https://www.jcu.edu.au/">
    <img alt="James Cook University" src="images/logos/JCU.jpg" height="100">
</a>

<a href="https://research.jcu.edu.au/portfolio/marcus.sheaves/">
    <img alt="Estuary and Coastal Wetland Ecosystems" src="images/logos/ECWE.jpeg" height="100">
</a>

<a href="https://www.jcu.edu.au/sicem">
    <img alt="Science Integrated Coastal Ecosystem Management" src="images/logos/SICEM.png" height="100">
</a>
<br>
<a href="https://www.ecolsoc.org.au/awards-and-prizes/holsworth-wildlife-research-endowment">
    <img alt="Holsworth Wildlife Research Endowment" src="images/logos/ESA_logo.png" height="75">
</a>

<a href="https://www.ecolsoc.org.au/awards-and-prizes/holsworth-wildlife-research-endowment">
    <img alt="Holsworth Wildlife Research Endowment" src="images/logos/Holsworth.png" height="75">
</a>
<br>
<a href="https://nectar.org.au/">
    <img alt="The National eResearch Collaboration Tools and Resources project" src="images/logos/nectardirectorate-logo.png" height="50">
</a>

<a href="https://www.qcif.edu.au/">
    <img alt="The Queensland Cyber Infrastructure Foundation" src="images/logos/qcif.png" height="50">
</a>

<a href="https://www.qriscloud.org.au/">
    <img alt="QRIScloud" src="images/logos/qris-logo.png" height="50">
</a>
<br>
<a href="https://www.mdatatechjcu.com/">
    <img alt="JCU's Marine Data Technology Hub" src="images/logos/Mdatatech.PNG" height="50">
</a>

<a href="https://www.tropwater.com/"> 
    <img alt="JCU's TropWATER" src="images/logos/TropWATER.jpg" height="50">
</a>
<br>

[ASPP-APAC 2019](https://scipy-school.org/)
<br>

<!-- Created by Cesar Herrera -->
[![alt text][1.2]][1]
[![alt text][2.2]][2]
<!-- icons without padding -->
[1.2]: images/logos/twittericon2.png (icon without padding)
[2.2]: images/logos/githubicon2.png (github icon without padding)
<!-- links to accounts -->
[1]: http://www.twitter.com/CexyNature
[2]: http://www.github.com/CexyNature
<!-- End -->