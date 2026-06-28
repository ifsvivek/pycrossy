"""Asset path resolution.

All visual/audio assets live in this project's bundled ``assets/`` directory. They were
originally created for an MIT-licensed project (© Evan Bacon) and are reused here under
that licence — see ``assets/ATTRIBUTION.md``. This module centralises path lookup and
defines the model and audio registries used by the rest of the game.
"""
from __future__ import annotations

import os
from typing import Dict

# pycrossy/ -> project root -> assets/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(_ROOT, "assets")
MODELS_DIR = os.path.join(ASSETS_DIR, "models")
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")

RETRO_FONT = os.path.join(FONTS_DIR, "retro.ttf")
TITLE_IMAGE = os.path.join(IMAGES_DIR, "title.png")


def model(*parts: str) -> str:
    return os.path.join(MODELS_DIR, *parts)


def audio(name: str) -> str:
    return os.path.join(AUDIO_DIR, name)


def image(*parts: str) -> str:
    return os.path.join(IMAGES_DIR, *parts)


def _mt(obj: str, png: str) -> Dict[str, str]:
    """A model+texture pair, returned as a ``{model, texture}`` mapping."""
    return {"model": obj, "texture": png}


# --- environment ---------------------------------------------------------
ENVIRONMENT = {
    "grass": {
        "0": _mt(model("environment/grass/model.obj"), model("environment/grass/light-grass.png")),
        "1": _mt(model("environment/grass/model.obj"), model("environment/grass/dark-grass.png")),
    },
    "road": {
        "0": _mt(model("environment/road/model.obj"), model("environment/road/stripes-texture.png")),
        "1": _mt(model("environment/road/model.obj"), model("environment/road/blank-texture.png")),
    },
    "log": {str(i): _mt(model(f"environment/log/{i}/0.obj"), model(f"environment/log/{i}/0.png")) for i in range(4)},
    "tree": {str(i): _mt(model(f"environment/tree/{i}/0.obj"), model(f"environment/tree/{i}/0.png")) for i in range(4)},
    "lily_pad": _mt(model("environment/lily_pad/0.obj"), model("environment/lily_pad/0.png")),
    "river": _mt(model("environment/river/0.obj"), model("environment/river/0.png")),
    "railroad": _mt(model("environment/railroad/0.obj"), model("environment/railroad/0.png")),
    "train_light": {
        "active": {
            "0": _mt(model("environment/train_light/active/0/0.obj"), model("environment/train_light/active/0/0.png")),
            "1": _mt(model("environment/train_light/active/1/0.obj"), model("environment/train_light/active/1/0.png")),
        },
        "inactive": _mt(model("environment/train_light/inactive/0.obj"), model("environment/train_light/inactive/0.png")),
    },
    "boulder": {str(i): _mt(model(f"environment/boulder/{i}/0.obj"), model(f"environment/boulder/{i}/0.png")) for i in range(2)},
}

# --- vehicles ------------------------------------------------------------
_CARS = ["police_car", "blue_car", "blue_truck", "green_car", "orange_car", "purple_car", "red_truck", "taxi"]
VEHICLES = {
    "train": {
        part: _mt(model(f"vehicles/train/{part}/0.obj"), model(f"vehicles/train/{part}/0.png"))
        for part in ("front", "middle", "back")
    },
    **{name: _mt(model(f"vehicles/{name}/0.obj"), model(f"vehicles/{name}/0.png")) for name in _CARS},
}
CAR_NAMES = list(_CARS)

# --- characters (only chicken is enabled by default) ---------------------
CHARACTERS = {
    "chicken": _mt(model("characters/chicken/0.obj"), model("characters/chicken/0.png")),
}
# Other rigs exist on disk; expose them so the character picker has variety.
for _name, _path in (
    ("bacon", "bacon/bacon"),
    ("brent", "brent/0"),
    ("avocoder", "avocoder/avocoder"),
    ("wheeler", "wheeler/wheeler"),
    ("palmer", "palmer/palmer"),
    ("juwan", "juwan/juwan"),
):
    _obj = model(f"characters/{_path}.obj")
    _png = model(f"characters/{_path}.png")
    if os.path.exists(_obj) and os.path.exists(_png):
        CHARACTERS[_name] = _mt(_obj, _png)


# --- audio registry ------------------------------------------------------
AUDIO = {
    "chicken": {
        "move": {str(i): audio(f"buck{i + 1}.wav") for i in range(12)},
        "die": {"0": audio("chickendeath.wav"), "1": audio("chickendeath2.wav")},
    },
    "car": {
        "passive": {"0": audio("car-engine-loop-deep.wav"), "1": audio("car-horn.wav")},
        "die": {"0": audio("carhit.mp3"), "1": audio("carsquish3.wav")},
    },
    "button_in": audio("Pop_1.wav"),
    "button_out": audio("Pop_2.wav"),
    "banner": audio("bannerhit3-g.wav"),
    "water": audio("watersplashlow.mp3"),
    "train_alarm": audio("Train_Alarm.wav"),
    "train": {
        "move": {"0": audio("train_pass_no_horn.wav"), "1": audio("train_pass_shorter.wav")},
        "die": {"0": audio("trainsplat.wav")},
    },
    "coin": audio("Get Coin 75 wav.wav"),
}
