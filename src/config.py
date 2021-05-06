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
c["AUDIO_TARGET_LEN_S"] = 5

# parameters for melspectrogram computation

c["MSG_N_FFT"] = 2048

c["MSG_TARGET_SIZE"] = {"time": 256, "freqs": 256}

c["MSG_N_HOP_LENGTH"] = (
    c["AUDIO_SR"] * c["AUDIO_TARGET_LEN_S"] // (c["MSG_TARGET_SIZE"]["time"] - 1)
)

c["MSG_N_MELS"] = c["MSG_TARGET_SIZE"]["freqs"]

c["MSG_NORMALIZE"] = True

# torchaudio melspectrogram device: cpu or cuda
c["TORCH_MELSPECTROGRAM_DEVICE"] = "cpu"

# normalize audio
c["AUDIO_NORMALIZE"] = True

# number of bins for latitude/longitude
c["GEO_COORDINATES_BINS"] = 18
