# vexcode

A set of functions for manipulating [VEXcode's](https://vr.vex.com) virtual robot 🤖.

Additionally, a few VEXcode scripts - found within the [scripts](scripts/) directory -
are in this repo.

This project started during the summer of 2022, made possible by and done in conjunction
with the wonderful students at
[Mariam Boyd Elementary School](https://www.warrenk12nc.org/o/mbes). For students or
staff interested in learning more, please visit
[go.ncsu.edu/vexcode](https://go.ncsu.edu/vexcode) (this webpage is also located within
this repo [here](web/index.html)).

## Quickstart

To run the Python code found within the [src](src/) directory, do the following:

1. Install Python 3.10; varies from system to system, but if you're on macOS, you can
   run `brew install python@3.10`, though this of course requires `homebrew`.
2. Install [poetry](https://python-poetry.org/), a Python package manager.
3. Run `poetry install` to install the required dependencies.
4. Done!

## [`generate_points.py`](src/generate_points.py)

This file is responsible for generating a traceable path for the VEX robot given an
arbitrary image. This works best with comparatively simple images, usually with very
clearly defined subject matter, i.e., one main subject in frame.
