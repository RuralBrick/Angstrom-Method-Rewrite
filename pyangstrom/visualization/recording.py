import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import Animation, FuncAnimation


def plot_recording(ax: Axes, df_recording: pd.DataFrame) -> Axes:
    ax.imshow(df_recording['Samples'].sum())
    ax.set_title(df_recording['Filename'].unique().item())
    ax.set_xlabel("x-coordinate (pixels)")
    ax.set_ylabel("y-coordinate (pixels)")
    return ax

def animate_recording(df_recording: pd.DataFrame) -> Animation:
    ax: Axes = None
    fig, ax = plt.subplots()
    arr_temp = np.stack(df_recording['Samples'])
    img = ax.imshow(
        df_recording['Samples'].iloc[0],
        vmin=arr_temp.min(),
        vmax=arr_temp.max(),
    )
    ax.set_title(df_recording.index[0].strftime('(%b %d) %H:%M:%S.%f'))
    ax.set_xlabel("x-coordinate (pixels)")
    ax.set_ylabel("y-coordinate (pixels)")
    def update(samples):
        time, frame = samples
        img.set_data(frame)
        ax.set_title(time.strftime('(%b %d) %H:%M:%S.%f'))
    interval = 1e3 * (
        df_recording.index
                    .to_series()
                    .diff()
                    .mode()
                    .item()
                    .total_seconds()
    )
    anim = FuncAnimation(
        fig,
        update,
        df_recording['Samples'].items(),
        interval=interval,
        repeat=False,
        cache_frame_data=False,
    )
    return anim
