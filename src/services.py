from src.config import c
from src.msg_provider import MSG_Provider
from src.wave_provider import WaveProvider

# services cache
_services = {}


def get_wave_provider(
    config=c,
    key=None,
) -> WaveProvider:

    if key is None:
        key = "wave_provider"

    if key not in _services:

        _services[key] = WaveProvider(
            audio_sr=config["AUDIO_SR"],
            cache_dir=config["CACHE_DIR"],
            src_dirs=config["SRC_DATA_DIRS"],
            normalize=config["AUDIO_NORMALIZE"],
            warn_on_silence=config["AUDIO_QUALITY_WARNINGS"],
        )

    return _services[key]


def get_msg_provider(
    config=c,
    n_mels=256,
    time_steps=256,
    key=None,
) -> MSG_Provider:

    if key is None:
        key = f"msg_provider:n_mels={n_mels}:time_steps={time_steps}"

    if key not in _services:

        _services[key] = MSG_Provider(
            n_mels=n_mels,
            target_n_mels=n_mels,
            time_steps=time_steps,
            n_fft=config["MSG_N_FFT"],
            power=config["MSG_POWER"],
            target_time_steps=time_steps,
            sample_rate=config["AUDIO_SR"],
            normalize=config["MSG_NORMALIZE"],
            f_min=config["MSG_FREQ_RANGE"][0],
            f_max=config["MSG_FREQ_RANGE"][1],
            audio_len_seconds=config["AUDIO_TARGET_LEN_S"],
            device=config["TORCH_MELSPECTROGRAM_DEVICE"],
        )

    return _services[key]
