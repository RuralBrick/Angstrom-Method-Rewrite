from pathlib import Path
from typing import get_type_hints

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import Animation, FuncAnimation

from pyangstrom.wrappers.data_extraction import (
    FrameConfig,
    get_first_frame,
    get_frame,
    iter_frame_path,
)


def animate_frames(
        working_directory: Path | str,
        dict_config: FrameConfig,
) -> Animation:
    p_wd = Path(working_directory)
    fig, ax = plt.subplots()
    img = ax.imshow(get_first_frame(p_wd, dict_config))
    def update(p_frame):
        img.set_data(get_frame(p_frame))
        ax.set_title(p_frame.stem)
        return img,
    return FuncAnimation(
        fig,
        update,
        iter_frame_path(p_wd / 'temperature data' / dict_config['rec_name']),
        interval=50,
        repeat=False,
        cache_frame_data=False,
    )

def animate_parameter_data(
        working_directory: Path | str,
        parameters_file: str,
) -> list[Animation]:
    p_wd = Path(working_directory)
    df_config = pd.read_csv(
        p_wd / 'batch process information' / parameters_file,
        usecols=get_type_hints(FrameConfig).keys(),
        dtype=get_type_hints(FrameConfig),
    )
    return [
        animate_frames(p_wd, dict(rec_name=rec_name))
        for rec_name in df_config['rec_name'].unique()
    ]
