# testing config

from src.config import c as prod_config

c = prod_config.copy()

c["SRC_DATA_DIRS"] = ["/app/tests/res"]
c["CACHE_DIR"] = "/tmp/cache"
