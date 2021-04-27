from src.services.data_provider import DataProvider
from src.config import c

# services cache
_services = {}


def get_data_provider(config=c) -> DataProvider:

    if "DataProvider" not in _services:
        _services["DataProvider"] = DataProvider(
            src_dirs=config["SRC_DATA_DIRS"],
            cache_dir=config["CACHE_DIR"],
            audio_sr=config["AUDIO_SR"],
        )

    return _services["DataProvider"]
