#!/bin/bash

# upload code as kaggle dataset

DATASET_ID="bc21-code"
GIT_URL="git@github.com:myurasov/Kaggle-BirdCLEF-2021.git"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
REPO_DIR="${THIS_DIR}/__repo__"

rm -rf "${REPO_DIR}"
mkdir -pv "${REPO_DIR}"

cd "${REPO_DIR}"
git clone "${GIT_URL}" .
rm -rf .git

kaggle datasets init
sed -i "s/INSERT_TITLE_HERE/${DATASET_ID}/" dataset-metadata.json 
sed -i "s/INSERT_SLUG_HERE/${DATASET_ID}/" dataset-metadata.json 

# second option should be used afer initial creation
# kaggle datasets create --dir-mode zip
kaggle datasets version --dir-mode zip -m "`date -u | sed 's/[ :]/_/g'`"

rm -rf "${REPO_DIR}"
