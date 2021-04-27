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

# audio sampling rate
c["AUDIO_SR"] = 32000

# audio sample length
c["AUDIO_TARTGET_LEN_S"] = 5

# parameters for melspectrogram computation

c["MSG_N_FFT"] = 2048

c["MSG_TARGET_SIZE"] = {"w": 224, "h": 224}

c["MSG_N_HOP_LENGTH"] = (
    c["AUDIO_SR"] // c["MSG_TARGET_SIZE"]["w"] * c["AUDIO_TARTGET_LEN_S"]
)

c["MSG_N_MELS"] = c["MSG_TARGET_SIZE"]["h"]

# torchaudio melspectrogram device: cpu or cuda
c["TA_MELSPECTROGRAM_DEVICE"] = "cpu"
