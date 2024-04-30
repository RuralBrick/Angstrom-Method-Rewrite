import logging
from pathlib import Path
import json

import yaml
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from matplotlib.animation import Animation

from pyangstrom.config import Config


logger = logging.getLogger('io')

def load_config(p_config: Path) -> Config:
    """
    Exceptions
    ----------
    ValueError
        File extension not recognized.
    """
    match p_config.suffix:
        case '.json':
            with open(p_config) as f:
                return json.load(f)
        case '.yaml' | '.yml':
            with open(p_config) as f:
                return yaml.safe_load(f)
        case _:
            msg = f"File extension not recognized: {p_config.suffix}"
            raise ValueError(msg)

def save_config(
        config: Config,
        p_directory: Path,
        name: str,
        file_extension='json',
):
    """
    Exceptions
    ----------
    ValueError
        Invalid file extension.
    """
    p_file = p_directory / f'{name}.{file_extension}'
    match file_extension:
        case 'json':
            with open(p_file, 'w') as f:
                json.dump(config, f, indent=4)
        case 'yaml' | 'yml':
            with open(p_file, 'w') as f:
                yaml.safe_dump(config, f)
        case _:
            raise ValueError(f"Invalid file extension: {file_extension}")
    logger.info(f"Saved config to {p_file}")

def iter_recording_path(p_rec: Path):
    return sorted(p_rec.iterdir(), key=lambda p: int(p.stem.split('_')[-1]))

def load_recording_frames(p_rec: Path):
    path_iterable = iter_recording_path(p_rec)
    if logger.getEffectiveLevel() <= logging.INFO:
        path_iterable = tqdm(path_iterable, total=len(list(p_rec.iterdir())))
    for p_frame in path_iterable:
        dict_frame = {}
        with open(p_frame) as f:
            for i, line in enumerate(f):
                line = line.rstrip()
                if line == '':
                    header_end = i + 1
                    break
                key, value = line.split(' = ')
                dict_frame[key] = value
        arr = np.genfromtxt(
            p_frame,
            dtype=np.float32,
            delimiter=',',
            skip_header=header_end,
        )
        logger.debug(f"{dict_frame=}, {arr.shape=}")
        dict_frame['Samples'] = arr
        yield dict_frame

def load_recording_csv(p_rec: Path) -> pd.DataFrame:
    logger.debug(f"Loading {p_rec}")
    df = pd.DataFrame.from_records(load_recording_frames(p_rec))
    df['Time'] = pd.to_datetime(df['Time'], format='%j:%H:%M:%S.%f')
    df = df.set_index('Time')
    return df

def recording_cache_exists(p_cache: Path, name: str) -> bool:
    p_file = p_cache / f'{name}.pkl.gz'
    return p_file.is_file()

def save_recording_cache(df_rec: pd.DataFrame, p_cache: Path, name: str):
    p_file = p_cache / f'{name}.pkl.gz'
    df_rec.to_pickle(p_file)
    logger.info(f"Saved recording cache to {p_file}")

def load_recording_cache(p_cache: Path, name: str) -> pd.DataFrame:
    p_file = p_cache / f'{name}.pkl.gz'
    logger.debug(f"Loading {p_file}")
    df = pd.read_pickle(p_file)
    return df

def save_animation(anim: Animation, p_directory: Path, name: str):
    p_file = p_directory / f'{name}.mp4'
    anim.save(p_file, dpi=300)
