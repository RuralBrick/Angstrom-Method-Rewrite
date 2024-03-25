import logging

import numpy as np
import pandas as pd


logger = logging.getLogger('io')

def iter_recording_path(p_rec):
    return sorted(p_rec.iterdir(), key=lambda p: int(p.stem.split('_')[-1]))

def load_recording_frames(p_rec):
    for p_frame in iter_recording_path(p_rec):
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
        dict_frame['Samples'] = arr
        logger.debug(f"{dict_frame=}, {arr.shape=}")
        yield dict_frame

def load_recording_csv(p_rec):
    logger.debug(f"Loading {p_rec}")
    df = pd.DataFrame.from_records(load_recording_frames(p_rec))
    df['Time'] = pd.to_datetime(df['Time'], format='%j:%H:%M:%S.%f')
    df = df.set_index('Time')
    return df

def save_recording_cache(df_rec, p_cache, name):
    p_file = p_cache / f'{name}.pkl.gz'
    df_rec.to_pickle(p_file)
    logger.info(f"Saved recording cache to {p_file}")

def load_recording_cache(p_cache, name):
    p_file = p_cache / f'{name}.pkl.gz'
    logger.debug(f"Loading {p_file}")
    df = pd.read_pickle(p_file)
    return df
