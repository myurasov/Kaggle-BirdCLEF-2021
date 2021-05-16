from src.config import c
from src.msg_provider import MSG_Provider
from src.wave_provider import WaveProvider

# services cache
services = {}


def get_wave_provider(
    config=c,
    key=None,
) -> WaveProvider:

    if key is None:
        key = "wave_provider"

    if key not in services:

        services[key] = WaveProvider(
            audio_sr=config["AUDIO_SR"],
            cache_dir=config["CACHE_DIR"],
            src_dirs=config["SRC_DATA_DIRS"],
            normalize=config["AUDIO_NORMALIZE"],
            cache_fragments=config["CACHE_AUDIO_FRAGMENTS"],
            warn_on_silence=config["AUDIO_QUALITY_WARNINGS"],
        )

    return services[key]


def get_msg_provider(
    config=c,
    key=None,
) -> MSG_Provider:

    if key is None:
        key = "msg_provider"

    if key not in services:

        services[key] = MSG_Provider(
            n_fft=config["MSG_N_FFT"],
            sample_rate=config["AUDIO_SR"],
            normalize=config["MSG_NORMALIZE"],
            f_min=config["MSG_FREQ_RANGE"][0],
            f_max=config["MSG_FREQ_RANGE"][1],
            audio_len_seconds=config["AUDIO_TARGET_LEN_S"],
            device=config["TORCH_MELSPECTROGRAM_DEVICE"],
        )

    return services[key]
