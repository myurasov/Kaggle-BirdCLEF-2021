#!/bin/bash

COMPETITION_DATA=`python -c "from src.config import c; print(c['COMPETITION_DATA'])"`

# competion data

DST_DIR="${DATA_DIR}"

if [ ! -d $DST_DIR ]
then
    mkdir -pv $DST_DIR
    cd $DST_DIR
    kaggle competitions download -c birdclef-2021
    unzip birdclef-2021.zip
    rm birdclef-2021.zip
fi
