import unittest

from src.config_test import c
from src.services import get_wave_provider


class TestWaveProvider(unittest.TestCase):
    def setUp(self):
        self.dp = get_wave_provider(c)

    def test_get_audio_fragment(self):
        # 16.7 second clip

        wave = self.dp.get_audio_fragment("XC11209.ogg")
        self.assertEqual(len(wave), 532933)

        wave = self.dp.get_audio_fragment("XC11209.ogg", 10, 15)
        self.assertEqual(len(wave), c["AUDIO_SR"] * 5)

        with self.assertRaises(Exception) as context:
            wave = self.dp.get_audio_fragment("XC11209.ogg", 0, 50)
        self.assertGreater(str(context.exception).find("Range"), -1)
        self.assertGreater(str(context.exception).find("doesn't exist"), -1)

    def test_get_audio_duration(self):
        res = self.dp.get_audio_duration("XC11209.ogg")
        self.assertEqual(res, 16.65415625)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
