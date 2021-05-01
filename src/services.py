from src.config import c
from src.data_provider import DataProvider
from src.msg_maker import MSG_Maker

# services cache
_services = {}


def get_data_provider(
    config=c,
    key="data_provider",
) -> DataProvider:

    if key not in _services:

        _services[key] = DataProvider(
            audio_sr=config["AUDIO_SR"],
            cache_dir=config["CACHE_DIR"],
            src_dirs=config["SRC_DATA_DIRS"],
            normalize=config["AUDIO_NORMALIZE"],
        )

    return _services[key]


def get_msg_maker(
    config=c,
    key="msg_maker",
) -> MSG_Maker:

    if key not in _services:

        _services[key] = MSG_Maker(
            n_fft=config["MSG_N_FFT"],
            n_mels=config["MSG_N_MELS"],
            sample_rate=config["AUDIO_SR"],
            hop_length=config["MSG_N_HOP_LENGTH"],
            device=config["TORCH_MELSPECTROGRAM_DEVICE"],
            target_msg_mels=c["MSG_TARGET_SIZE"]["freqs"],
            target_msg_time_steps=c["MSG_TARGET_SIZE"]["time"],
        )

    return _services[key]
