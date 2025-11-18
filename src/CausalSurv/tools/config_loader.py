import tomllib
from pathlib import Path

def load_config(config):
    if isinstance(config, dict):
        return config
    elif isinstance(config, (str, Path)):
        with open(config, "rb") as f:
            return tomllib.load(f)
    else:
        raise ValueError("cell_config must be a dict or a path to a .toml file")