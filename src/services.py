from src.config import c
from src.msg_provider import MSG_Provider
from src.wave_provider import WaveProvider

# services cache
_services = {}


def get_wave_provider(
    config=c,
    key="wave_provider",
) -> WaveProvider:

    if key not in _services:

        _services[key] = WaveProvider(
            audio_sr=config["AUDIO_SR"],
            cache_dir=config["CACHE_DIR"],
            src_dirs=config["SRC_DATA_DIRS"],
            normalize=config["AUDIO_NORMALIZE"],
        )

    return _services[key]


def get_msg_provider(
    config=c,
    n_mels=256,
    time_steps=256,
    key="msg_maker",
) -> MSG_Provider:

    if key not in _services:

        _services[key] = MSG_Provider(
            n_mels=n_mels,
            target_n_mels=n_mels,
            time_steps=time_steps,
            n_fft=config["MSG_N_FFT"],
            target_time_steps=time_steps,
            sample_rate=config["AUDIO_SR"],
            normalize=config["MSG_NORMALIZE"],
            audio_len_seconds=c["AUDIO_TARGET_LEN_S"],
            device=config["TORCH_MELSPECTROGRAM_DEVICE"],
        )

    return _services[key]
