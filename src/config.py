import os

# we're on kaggle
is_kaggle = "KAGGLE_CONTAINER_NAME" in os.environ

c = {}

c["SEED"] = 123

# input data dir
c["DATA_DIR"] = "/app/_data"
#
if is_kaggle:
    c["DATA_DIR"] = "/kaggle/input/birdclef-2021"

# where to put generated data
c["WORK_DIR"] = "/app/_work"
#
if is_kaggle:
    c["WORK_DIR"] = "/tmp/_work"

# cache directory
c["CACHE_DIR"] = c["WORK_DIR"] + "/cache"

# list of source data dirs (globs can be used)
c["SRC_DATA_DIRS"] = [
    c["DATA_DIR"] + "/competition_data/train_short_audio/*",
    c["DATA_DIR"] + "/competition_data/train_soundscapes",
]
#
if is_kaggle:
    c["SRC_DATA_DIRS"] = [
        c["DATA_DIR"] + "/train_short_audio/*",
        c["DATA_DIR"] + "/train_soundscapes",
    ]

# audio params
c["AUDIO_SR"] = 32000
c["AUDIO_NORMALIZE"] = True
c["AUDIO_TARGET_LEN_S"] = 5
c["AUDIO_QUALITY_WARNINGS"] = False

# parameters for melspectrogram computation
c["MSG_N_FFT"] = 2048
c["MSG_NORMALIZE"] = False  # -mean, /std
c["MSG_FREQ_RANGE"] = [0, 16000]
c["MSG_POWER"] = 3.0

# torchaudio melspectrogram device: cpu or cuda
c["TORCH_MELSPECTROGRAM_DEVICE"] = "cpu"

# number of bins for latitude/longitude
c["GEO_COORDINATES_BINS"] = 18

# columns that should be present in the DF that can be joined into a dataset
c["DATASET_COLS"] = [
    "filename",
    "_primary_labels",
    "_secondary_labels",
    "_from_s",
    "_to_s",
    "_year",
    "_month",
    "latitude",
    "longitude",
    "rating",
    "_source",
]

# files to exclude from the dataset
c["EXCLUDE_FILES"] = ["XC579430.ogg", "XC590621.ogg"]
