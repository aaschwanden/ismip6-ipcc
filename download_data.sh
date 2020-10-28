#!/bin/bash

file=v7_CMIP5_pub.zip
# Download ISMIP6 scalars
wget -nc https://zenodo.org/record/3939037/files/$file
unzip $file
