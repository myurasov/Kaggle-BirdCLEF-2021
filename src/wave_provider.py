import os
from glob import glob
from hashlib import md5
import warnings

import librosa
import numpy as np


class WaveProvider:
    def __init__(
        self,
        src_dirs=[],
        cache_dir=None,
        audio_sr=32000,
        normalize=False,
    ):
        self._audio_sr = audio_sr
        self._cache_dir = cache_dir
        self._normalize = normalize
        self._src_dirs = self._resolve_src_dir_globs(src_dirs)

    def _resolve_src_dir_globs(self, src_dirs):
        """Resolve globs in src_dirs"""

        res = []

        for src_dir_or_glob in src_dirs:
            for dir in glob(src_dir_or_glob, recursive=True):
                if os.path.isdir(dir):
                    res.append(dir)

        return res

    def _find_src_file(self, file_name):
        """Look for file in one of the source dirs"""

        for src_dir in self._src_dirs:
            file_path = os.path.join(src_dir, file_name)
            if os.path.isfile(file_path):
                return os.path.normpath(file_path)

        raise FileNotFoundError(f"Can't find {file_name} in one of the source dirs.")

    def get_audio_duration(self, file_name):
        """Get audio file duration [second]"""

        file_path = (
            file_name
            if (os.path.sep == file_name[0])
            else self._find_src_file(file_name)
        )

        return librosa.get_duration(filename=file_path)

    def get_audio_fragment(self, file_name, range_seconds=None):
        """Get audio fragment by file name"""

        file_path = self._find_src_file(file_name)

        wave = None
        cache_key = ""
        cache_dir = ""
        cache_path = ""

        range_samples = (
            None
            if range_seconds is None
            else [
                int(self._audio_sr * range_seconds[0]),
                int(self._audio_sr * range_seconds[1]),
            ]
        )

        # try reading from cache
        if self._cache_dir is not None:
            cache_key = f"{file_path}:sr={self._audio_sr}:{range_samples=}"
            cache_key = md5(cache_key.encode()).hexdigest()
            cache_dir = os.path.join(self._cache_dir, "audio_fragments")
            cache_dir = os.path.join(cache_dir, cache_key[0], cache_key[1])
            cache_path = os.path.join(cache_dir, cache_key)
            if os.path.isfile(cache_path):
                wave = np.load(cache_path + ".npy")

        if wave is None:
            # load

            if range_seconds is None:
                # read from actual file only when no range is specified
                wave, sr = librosa.load(file_path, sr=self._audio_sr)
                assert sr == self._audio_sr

                # normalize when reading from the whole file
                if self._normalize:
                    assert wave.dtype == np.float32

                    wave -= np.mean(wave)

                    std = np.std(wave)
                    if std != 0:
                        wave /= np.std(wave)
                    else:
                        warnings.warn(f'SDT=0 in "{file_path}"', UserWarning)

            else:
                # read from a possibly cached whole file
                wave = self.get_audio_fragment(file_name=file_name, range_seconds=None)

            # crop
            if range_samples is not None:
                wave = wave[range_samples[0] : range_samples[1]]

            # check if the asked range is valid
            if range_samples is not None:
                if len(wave) != range_samples[1] - range_samples[0]:
                    raise Exception(
                        f"Range {range_seconds[0]}-{range_seconds[1]} doesn't exist"
                        + f' in file "{file_name}"'
                    )

            wave = wave.astype(np.float16)

            # save to cache
            if self._cache_dir is not None:
                if not os.path.isdir(cache_dir):
                    os.makedirs(cache_dir)
                np.save(cache_path, wave)

        return wave
