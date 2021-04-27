c = {}

c["SEED"] = 123

# input data dir
c["DATA_DIR"] = "/app/_data"

# where to put generated data
c["WORK_DIR"] = "/app/_work"

# cache directory
c["CACHE_DIR"] = c["WORK_DIR"] + "/cache"

# list of source data dirs (globs can be used)
c["SRC_DATA_DIRS"] = [
    c["DATA_DIR"] + "/competition_data/train_short_audio/*",
    c["DATA_DIR"] + "/competition_data/train_soundscapes",
]

# default audio sampling rate
c["AUDIO_SR"] = 32000
