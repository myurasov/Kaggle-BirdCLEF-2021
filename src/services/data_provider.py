import os
from glob import glob


class DataProvider:
    def __init__(
        self,
        src_dirs=[],
        cache_dir=None,
        audio_sr=32000,
    ):
        self._audio_sr = audio_sr
        self._cahce_dir = cache_dir
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

    def get_audio_fragment(self, file_name, start_sec=None, end_sec=None):
        """Get audio fragment by file name"""

        file_path = self._find_src_file(file_name)
        return file_path
