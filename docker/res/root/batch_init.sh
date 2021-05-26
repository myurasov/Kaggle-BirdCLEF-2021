#!/bin/bash

cd ~
git clone --recursive git@github.com:myurasov/Kaggle-BirdCLEF-2021.git repo
cp -r repo/* /app/

cd /app

# competion data
cli/download_data.sh

# convert data from natasha
mkdir -p _work
jupyter nbconvert --to notebook --execute notebooks/convert_n_data.ipynb
