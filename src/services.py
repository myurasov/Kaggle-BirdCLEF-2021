from src.config import c
from src.data_provider import DataProvider
from torchaudio.transforms import MelSpectrogram

# services cache
_services = {}


def get_data_provider(
    config=c,
    key="data_provider",
) -> DataProvider:

    if key not in _services:

        _services[key] = DataProvider(
            src_dirs=config["SRC_DATA_DIRS"],
            cache_dir=config["CACHE_DIR"],
            audio_sr=config["AUDIO_SR"],
        )

    return _services[key]


def get_torch_melspectrogram(
    config=c,
    key="torch_melspectrogram",
) -> MelSpectrogram:

    if key not in _services:

        _services[key] = MelSpectrogram(
            power=2.0,
            center=True,
            norm="slaney",
            onesided=True,
            win_length=None,
            pad_mode="reflect",
            n_fft=config["MSG_N_FFT"],
            n_mels=config["MSG_N_MELS"],
            sample_rate=config["AUDIO_SR"],
            hop_length=config["MSG_N_HOP_LENGTH"],
        ).to(config["TA_MELSPECTROGRAM_DEVICE"])

    return _services[key]
