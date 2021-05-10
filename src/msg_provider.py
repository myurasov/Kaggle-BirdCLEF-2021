import warnings

import librosa
import numpy as np
import torch
from PIL import Image
from torchaudio.transforms import MelSpectrogram


class MSG_Provider:
    def __init__(
        self,
        n_fft,
        n_mels,
        time_steps,
        sample_rate,
        audio_len_seconds,
        target_n_mels,
        target_time_steps,
        f_min=0,
        f_max=None,
        power=2,
        normalize=False,
        device="cpu",
    ):
        if f_max is None:
            f_max = sample_rate // 2

        hop_length = sample_rate * audio_len_seconds // (time_steps - 1)

        self._msg_transform = MelSpectrogram(
            power=power,
            center=True,
            norm="slaney",
            onesided=True,
            win_length=None,
            pad_mode="reflect",
            n_fft=n_fft,
            n_mels=n_mels,
            sample_rate=sample_rate,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
        ).to(device)

        self._device = device
        self._normalize = normalize
        self._target_n_mels = target_n_mels
        self._target_time_steps = target_time_steps

    def msg(self, wave):
        wave = torch.tensor(wave.reshape([1, -1]).astype(np.float32)).to(self._device)

        msg = self._msg_transform(wave)[0].cpu().numpy()
        msg = librosa.power_to_db(msg)

        if self._normalize:
            assert msg.dtype == np.float32 or msg.dtype == np.float64
            msg -= np.mean(msg)
            std = np.std(msg)
            if std != 0:
                msg /= std

        # if melspectrogram size mismatches with target, resize it
        if msg.shape != (self._target_n_mels, self._target_time_steps):
            warnings.warn("MelSpectrogram is resized", UserWarning)
            msg = Image.fromarray(msg)
            msg = msg.resize(
                (self._target_time_steps, self._target_n_mels),
                Image.BICUBIC,
            )
            msg = np.array(msg)

        return msg.astype(np.float32)
