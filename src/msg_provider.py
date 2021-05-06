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
        hop_length,
        sample_rate,
        target_msg_mels,
        target_msg_time_steps,
        normalize=False,
        device="cpu",
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
        self._normalize = normalize
        self._target_msg_mels = target_msg_mels
        self._target_msg_time_steps = target_msg_time_steps

    def msg(self, wave):
        wave = torch.tensor(wave.reshape([1, -1]).astype(np.float32)).to(self._device)

        msg = self._msg_transform(wave)[0].cpu().numpy()
        msg = librosa.power_to_db(msg)

        if self._normalize:
            assert msg.dtype == np.float32
            msg -= np.mean(msg)
            msg /= np.mean(msg)

        # if melspectrogram size mismatches with target, resize it
        if msg.shape != (self._target_msg_mels, self._target_msg_time_steps):
            msg = Image.fromarray(msg)
            msg = msg.resize(
                (self._target_msg_time_steps, self._target_msg_mels),
                Image.BICUBIC,
            )
            msg = np.array(msg)

        return msg.astype(np.float16)
