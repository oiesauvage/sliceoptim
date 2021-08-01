# sliceoptim

`sliceoptim` is a Python package providing tools to optimize slicer settings for FDM 3D printers.  
Documentation is available [HERE](https://oiesauvage.github.io/sliceoptim/index.html).

## Description

The main objective of `sliceptim` is to automate the complex process of finding good slicing parameters for each FDM printer / filament pair, while saving time and plastic use.

In a nutshell, `sliceoptim` interfaces programmatically the FDM slicing software Slic3r to generate batches of samples featuring semi-random exploration of slicing parameters (speeds, extrusion rates...). Then, on the basis of ratings provided by the user and print time estimations, `sliceoptim` feeds a Gaussian Process model (implemented with the library Skopt) to find parameter values optimizing both print time and quality.

The "quality" measure can be of any kind, as soon as it is as consistent as possible. For example, the user can provide an esthetic aspect evaluation based on visual defects, which will result in improvement of prints appearance. Another use could be the realization of stress tests to improve robustness, the quality index can therefore be the negative of the maximum applicable force on samples.

A GUI for `sliceoptim` is available as an Octoprint plugin.

## Installation

To use `sliceoptim`, you must have `libslic3r` installed, as well as `Python 3.8+`. This library is not tested yet on Windows or Mac (only Linux), contributions are welcome. If you all these requirement, you can install the last commit with pip:

```bash
pip install git+https://github.com/oiesauvage/sliceoptim.git
```

for development purpose install `conda`, fork and clone this repository and run:

```bash
conda env create -f environment.yml
conda activate sliceoptim
python setup.py develop
```

## How to use

The process of optimizing slicing parameters for a printer / filament pair is called an `Experiment` and managed by the class of the same name. Such process can be summarized by the following steps:

1) Definition of printer and filament objects
2) Definition of the `Experiment` with the corresponding parametric space (parameters which will be optimized on given bounds)
3) Generation of G-Code for the new batch and printing
4) Registration of printed samples ratings in the `Experiment` object
5) Evaluate optimal parameters with corresponding uncertainty
6) Repeat from step 3 until satisfying results

Since an example worth thousand words, you will find illustrating notebooks in the examples folder.

## DISCLAMER ! 

> THIS SOFTWARE IS DELIVERED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND. ANY DAMAGE WHICH MAY OCCUR ON YOUR HARDWARE OR PEOPLE IS YOUR SOLE AND UNIQUE RESPONSIBILITY. SINCE THIS SOFTWARE WILL EXPLORE VARIOUS SLICING PARAMETERS, ALWAYS STAY PHYSICALLY CLOSE TO YOUR PRINTER IN ORDER TO ACT AS QUICK AS POSSIBLE IF NECESSARY.

## Licence

Copyright 2021 Nils ARTIGES

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

