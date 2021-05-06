from src.config import c
from src.wave_provider import WaveProvider
from src.msg_provider import MSG_Provider

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
    key="msg_maker",
) -> MSG_Provider:

    if key not in _services:

        _services[key] = MSG_Provider(
            n_fft=config["MSG_N_FFT"],
            n_mels=config["MSG_N_MELS"],
            sample_rate=config["AUDIO_SR"],
            normalize=config["MSG_NORMALIZE"],
            hop_length=config["MSG_N_HOP_LENGTH"],
            device=config["TORCH_MELSPECTROGRAM_DEVICE"],
            target_msg_mels=c["MSG_TARGET_SIZE"]["freqs"],
            target_msg_time_steps=c["MSG_TARGET_SIZE"]["time"],
        )

    return _services[key]
