# Cellori (Cell Origin)
A fast and robust algorithm for clustered nuclei segmentation.

## General
The Cellori algorithm segments nuclei by applying a Gaussian filter to smoothen out background noise, calculating local thresholds to isolate the foreground, and splitting clustered nuclei via local maxima analysis.

## Installation

Install Cellori from [PyPI](https://pypi.org/project/cellori/).

```
pip install cellori
```

## Usage

The built-in GUI is the easiest way to use the Cellori algorithm.

```python
from cellori import Cellori

# image: a np.ndarray array or a path to a ND2 or TIFF file
    # if image is the path to a ND2 or TIFF file with multiple channels, the parameter nuclei_channel can be specified as a kwarg
    # if image is the path to a ND2 file, StitchWell parameters nd2_overlap and nd2_stitch_channel can be specified as kwargs

Cellori(image,**kwargs).gui()
```

Functions for getting or saving nuclei coordinates without the GUI are also available.