from lib.utils import read_json
from tensorflow import keras
from tqdm import tqdm

from src.config import c
from src.generator import Generator
from src.models import (
    Div,
    Float2DToFloatRGB,
    Float2DToRGB,
    MelSpectrogram,
    PowerToDb,
    SinCos,
    YMToDate,
)
from src.services import get_msg_provider, get_wave_provider


def predict(
    model_path,
    df,
    quiet=False,
):
    keras.backend.clear_session()

    # metadata from model training run
    meta = read_json(model_path.replace(".h5", ".json"))

    # set kaggle-specific options from config saved with a model

    for k in [
        "WORK_DIR",
        "CACHE_DIR",
        "COMPETITION_DATA",
        "SRC_DATA_DIRS",
        "CACHE_AUDIO_FRAGMENTS",
    ]:
        meta["config"][k] = c[k]

    if "msg_power" in meta["args"]:
        c["MSG_POWER"] = meta["args"]["msg_power"]

    # load model
    model = keras.models.load_model(
        model_path,
        custom_objects={
            "SinCos": SinCos,
            "Div": Div,
            "YMToDate": YMToDate,
            "MelSpectrogram": MelSpectrogram,
            "Float2DToFloatRGB": Float2DToFloatRGB,
            "Float2DToRGB": Float2DToRGB,
            "PowerToDb": PowerToDb,
        },
    )

    # create generator

    wave_p = get_wave_provider(meta["config"])

    if meta["args"]["model"].startswith("msg_"):

        input_shape = model.get_layer("i_msg").input_shape[0][1:]
        msg_p = get_msg_provider(meta["config"])

        generator = Generator(
            df=df,
            shuffle=False,
            augmentation=None,
            rating_as_sw=False,
            rareness_as_sw=False,
            msg_provider=msg_p,
            wave_provider=wave_p,
            msg_output_size=input_shape,
            msg_power=meta["config"]["MSG_POWER"],
            geo_coordinates_bins=meta["config"]["GEO_COORDINATES_BINS"],
            batch_size=1,
        )

    else:

        generator = Generator(
            df=df,
            shuffle=False,
            augmentation=None,
            rating_as_sw=False,
            rareness_as_sw=False,
            msg_provider=None,
            wave_provider=wave_p,
            geo_coordinates_bins=meta["config"]["GEO_COORDINATES_BINS"],
            batch_size=1,
        )

    # predict
    Y_pred = model.predict(
        x=generator,
        verbose=0 if quiet else 1,
    )

    return Y_pred, meta["labels"]
