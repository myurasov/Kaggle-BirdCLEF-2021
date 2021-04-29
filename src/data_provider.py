import os
from glob import glob
from hashlib import md5

import librosa
import numpy as np


class DataProvider:
    def __init__(
        self,
        src_dirs=[],
        cache_dir=None,
        audio_sr=32000,
    ):
        self._audio_sr = audio_sr
        self._cache_dir = cache_dir
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

    def get_audio_fragment(self, file_name, start_s=0, end_s=None):
        """Get audio fragment by file name"""

        file_path = self._find_src_file(file_name)

        start_sample = int(start_s * self._audio_sr)
        end_sample = None if end_s is None else int(end_s * self._audio_sr)

        wave = None
        cache_key = ""
        cache_dir = ""
        cache_path = ""

        # try reading from cache
        if self._cache_dir is not None:
            cache_key = f"{file_path}:sr={self._audio_sr}:{start_sample=}:{end_sample=}"
            cache_key = md5(cache_key.encode()).hexdigest()
            cache_dir = os.path.join(self._cache_dir, "audio_fragments")
            cache_dir = os.path.join(cache_dir, cache_key[0], cache_key[1])
            cache_path = os.path.join(cache_dir, cache_key)
            if os.path.isfile(cache_path):
                wave = np.load(cache_path + ".npy")

        # read from actual file
        if wave is None:
            # load
            wave, sr = librosa.load(file_path, sr=self._audio_sr)
            wave = wave.astype(np.float16)
            assert sr == self._audio_sr

            # crop
            wave = wave[start_sample:end_sample]

            # check if the asked range is valid
            if (end_sample is not None) and (len(wave) != end_sample - start_sample):
                raise Exception(
                    f'Range {start_s}-{end_s} doesn\'t exist in file "{file_name}"'
                )

            # save to cache
            if self._cache_dir is not None:
                if not os.path.isdir(cache_dir):
                    os.makedirs(cache_dir)
                np.save(cache_path, wave)

        return wave
