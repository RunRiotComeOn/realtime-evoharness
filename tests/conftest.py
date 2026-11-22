import copy
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DRONE = ROOT / "examples" / "drone"
sys.path[:0] = [str(ROOT), str(DRONE), str(ROOT / "tests")]

from evoharness.config import Config  # noqa: E402


def drone_config(**over):
    data = yaml.safe_load((DRONE / "config.yaml").read_text())
    data["scenarios"] = {"train": ["a1", "a2", "a3"], "test": ["b1"]}
    data["proposer"] = {"backend": "python", "callable": "scripted:chat"}
    data["verifier"] = {"enabled": True}
    data["python_path"] = [".", str(ROOT / "tests")]
    data["loop"] = {"rounds": 1, "tools": {"trace": 0, "trial_episode": 0, "point_trial": 0}, "smoke_seconds": 3}
    data["parallel"] = 4

    def deep(d, o):
        for k, v in o.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                deep(d[k], v)
            else:
                d[k] = v
    deep(data, copy.deepcopy(over))
    return Config.from_dict(data, DRONE)


@pytest.fixture
def cfg():
    return drone_config()
