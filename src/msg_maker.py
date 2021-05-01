import librosa
import numpy as np
import torch
from torchaudio.transforms import MelSpectrogram

from PIL import Image


class MSG_Maker:
    def __init__(
        self,
        n_fft,
        n_mels,
        hop_length,
        sample_rate,
        device="cpu",
        target_msg_size=(256, 256),
    ):

        self._msg_transform = MelSpectrogram(
            power=2.0,
            center=True,
            norm="slaney",
            onesided=True,
            win_length=None,
            pad_mode="reflect",
            n_fft=n_fft,
            n_mels=n_mels,
            sample_rate=sample_rate,
            hop_length=hop_length,
        ).to(device)

        self._device = device
        self._target_msg_size = target_msg_size

    def msg(self, wave):
        wave = torch.tensor(wave.reshape([1, -1]).astype(np.float32)).to(self._device)

        msg = self._msg_transform(wave)[0].cpu().numpy()
        msg = librosa.power_to_db(msg)

        print(msg.shape)

        return msg
