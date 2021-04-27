import unittest

from src.services import get_data_provider
from src.config_test import c


class Test_DataProvider(unittest.TestCase):
    def setUp(self):
        self.dp = get_data_provider(c)

    def test_get_audio_fragment(self):
        res = self.dp.get_audio_fragment("XC11209.ogg")
        self.assertEqual(res, "/app/tests/res/XC11209.ogg")


if __name__ == "__main__":
    unittest.main(warnings="ignore")
