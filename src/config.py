import os

# we're on kaggle - needs to be set up manually
is_kaggle = "__KAGGLE__" in os.environ

c = {}

c["SEED"] = 123

# input data dir
c["DATA_DIR"] = "/app/_data"

# where to put generated data
c["WORK_DIR"] = "/app/_work"
if is_kaggle:
    c["WORK_DIR"] = "/tmp/_work"

# cache directory
c["CACHE_DIR"] = c["WORK_DIR"] + "/cache"

# competition data
c["COMPETITION_DATA"] = c["DATA_DIR"] + "/competition_data"
if is_kaggle:
    c["COMPETITION_DATA"] = "/kaggle/input/birdclef-2021"

# list of source data dirs (globs can be used)
c["SRC_DATA_DIRS"] = [
    c["COMPETITION_DATA"] + "/train_short_audio/*",
    c["COMPETITION_DATA"] + "/*_soundscapes",
]

# audio params
c["AUDIO_SR"] = 32000
c["AUDIO_NORMALIZE"] = True
c["AUDIO_TARGET_LEN_S"] = 5
c["AUDIO_QUALITY_WARNINGS"] = False

# whether to cache cut fragments, and not only entire waves
c["CACHE_AUDIO_FRAGMENTS"] = True
if is_kaggle:
    c["CACHE_AUDIO_FRAGMENTS"] = False

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
    "_extra_primary_labels",
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
c["EXCLUDE_FILES"] = [
    "XC579430.ogg",  # silence
    "XC590621.ogg",  # silence
    "XC359315.ogg",  # year 201
    "XC207317.ogg",  # year 201
    "XC493567.ogg",  # year 201
    "XC493567.ogg",  # year 201
    "XC600308.ogg",  # year 202
    "XC452323.ogg",  # year 199
    "XC204856.ogg",  # year 2104
]
