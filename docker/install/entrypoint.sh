#!/bin/bash

python -m pip install .
python -m pip show --verbose storch
python -m storch --verbose
rm -rf ./build ./storch.egg-info
