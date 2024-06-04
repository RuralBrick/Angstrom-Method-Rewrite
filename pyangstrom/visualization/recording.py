import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.animation import Animation, FuncAnimation


def plot_recording(ax: Axes, df_recording: pd.DataFrame) -> Axes:
    ax.imshow(df_recording['Samples'].sum())
    return ax

def animate_recording(df_recording: pd.DataFrame) -> Animation:
    fig, ax = plt.subplots()
    arr_temp = np.stack(df_recording['Samples'])
    img = ax.imshow(
        df_recording['Samples'].iloc[0],
        vmin=arr_temp.min(),
        vmax=arr_temp.max(),
    )
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
