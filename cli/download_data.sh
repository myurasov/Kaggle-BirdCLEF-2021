#!/bin/bash

COMPETITION_DATA=`python -c "from src.config import c; print(c['COMPETITION_DATA'])"`

# competion data

DST_DIR="${COMPETITION_DATA}"

if [ ! -d $DST_DIR ]
then
    mkdir -pv $DST_DIR
    cd $DST_DIR
    kaggle competitions download -c birdclef-2021
    unzip birdclef-2021.zip
    rm birdclef-2021.zip

    # mimic kaggle data layout
    cat train_soundscape_labels.csv | cut -d , -f 1-4 > fake_test.csv
fi
