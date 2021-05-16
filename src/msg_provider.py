import warnings

import librosa
import numpy as np
import torch
from PIL import Image
from torchaudio.transforms import MelSpectrogram


class MSG_Provider:
    def __init__(
        self,
        n_fft: int,
        sample_rate: int,
        audio_len_seconds: float,
        f_min: int,
        f_max: int,
        normalize: bool,
        device: str = "cpu",
    ):
        self._f_min = f_min
        self._f_max = f_max
        self._n_fft = n_fft
        self._device = device
        self._normalize = normalize
        self._sample_rate = sample_rate
        self._audio_len_seconds = audio_len_seconds

        self._cache = {}

    def _get_MelSpectrogram(self, power, n_mels, time_steps) -> MelSpectrogram:

        # cache torchaudio.transforms.Melspectrogram objects
        # for each set or parameters

        key = f"{power:.1f}:{n_mels:d}:{time_steps:d}"

        if key in self._cache:
            return self._cache[key]

        hop_length = self._sample_rate * self._audio_len_seconds // (time_steps - 1)

        self._cache[key] = MelSpectrogram(
            power=power,
            win_length=None,
            n_fft=self._n_fft,
            n_mels=n_mels,
            sample_rate=self._sample_rate,
            hop_length=hop_length,
            f_min=self._f_min,
            f_max=self._f_max,
        ).to(self._device)

        return self._cache[key]

    def msg(self, wave, n_mels, time_steps, power):
        wave = torch.tensor(wave.reshape([1, -1]).astype(np.float32)).to(self._device)

        transform = self._get_MelSpectrogram(
            n_mels=n_mels, time_steps=time_steps, power=power
        )

        msg = transform(wave)[0].cpu().numpy()
        msg = librosa.power_to_db(msg)

        if self._normalize:
            assert msg.dtype == np.float32 or msg.dtype == np.float64
            msg -= np.mean(msg)
            std = np.std(msg)
            if std != 0:
                msg /= std

        assert msg.shape == (n_mels, time_steps)
        return msg.astype(np.float32)
