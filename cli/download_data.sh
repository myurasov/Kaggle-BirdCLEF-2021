#!/bin/bash

DATA_DIR=`python -c "from src.config import c; print(c['DATA_DIR'])"`

# competion data

DST_DIR="${DATA_DIR}/competition_data"

if [ ! -d $DST_DIR ]
then
    mkdir -pv $DST_DIR
    cd $DST_DIR
    kaggle competitions download -c birdclef-2021
    unzip birdclef-2021.zip
    rm birdclef-2021.zip
fi