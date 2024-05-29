import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Wedge, Circle
import matplotlib.pyplot as plt
from matplotlib.animation import Animation, FuncAnimation

from pyangstrom.transform import (
    Direction,
    CartesianGeometry,
    PolarGeometry,
    Region,
    collapse_region,
)
from pyangstrom.signal import region_to_properties


def plot_recording(ax: Axes, df_recording: pd.DataFrame) -> Axes:
    ax.imshow(df_recording['Samples'].sum())
    return ax

def plot_cartesian_geometry(
        ax: Axes,
        df_recording: pd.DataFrame,
        geometry: CartesianGeometry,
) -> Axes:
    ax = plot_recording(ax, df_recording)
    ax.add_patch(Rectangle(
        (geometry['min_x_pixels'], geometry['min_y_pixels']),
        geometry['max_x_pixels'] - geometry['min_x_pixels'],
        geometry['max_y_pixels'] - geometry['min_y_pixels'],
        hatch='..',
        edgecolor='red',
        facecolor='none',
    ))
    if geometry['heat_source'] in Direction.LESSER_X:
        ax.plot(
            [geometry['min_x_pixels'], geometry['min_x_pixels']],
            [geometry['min_y_pixels'], geometry['max_y_pixels']],
            linewidth=2,
            color='blue',
        )
    elif geometry['heat_source'] in Direction.GREATER_X:
        ax.plot(
            [geometry['max_x_pixels'], geometry['max_x_pixels']],
            [geometry['min_y_pixels'], geometry['max_y_pixels']],
            linewidth=2,
            color='blue',
        )
    elif geometry['heat_source'] in Direction.LESSER_Y:
        ax.plot(
            [geometry['min_x_pixels'], geometry['max_x_pixels']],
            [geometry['min_y_pixels'], geometry['min_y_pixels']],
            linewidth=2,
            color='blue',
        )
    elif geometry['heat_source'] in Direction.GREATER_Y:
        ax.plot(
            [geometry['min_x_pixels'], geometry['max_x_pixels']],
            [geometry['max_y_pixels'], geometry['max_y_pixels']],
            linewidth=2,
            color='blue',
        )
    return ax

def plot_polar_geometry(
        ax: Axes,
        df_recording: pd.DataFrame,
        geometry: PolarGeometry,
) -> Axes:
    ax = plot_recording(ax, df_recording)
    ax.add_patch(Wedge(
        (geometry['center']['x_pixels'], geometry['center']['y_pixels']),
        geometry['max_r_pixels'],
        geometry['min_theta_degrees'],
        geometry['max_theta_degrees'],
        hatch='..',
        edgecolor='red',
        facecolor='none',
    ))
    ax.add_patch(Circle(
        (geometry['center']['x_pixels'], geometry['center']['y_pixels']),
        geometry['min_r_pixels'],
        linewidth=2,
        edgecolor='r',
        linestyle='solid',
        facecolor='none',
    ))
    ax.add_patch(Circle(
        (geometry['center']['x_pixels'], geometry['center']['y_pixels']),
        geometry['max_r_pixels'],
        linewidth=2,
        edgecolor='r',
        linestyle='solid',
        facecolor='none',
    ))
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

def animate_region(region: Region) -> Animation:
    fig, ax = plt.subplots()
    region = collapse_region(region)
    disp = region_to_properties(region).displacements_meters
    ln, = ax.plot(disp, region.temperatures_kelvin[0])
    ax.set_ylim(
        region.temperatures_kelvin.min(),
        region.temperatures_kelvin.max(),
    )
    def update(temps):
        ln.set_data(disp, temps)
    interval = 1e3 * (
        region.timestamps
              .to_series()
              .diff()
              .mode()
              .item()
              .total_seconds()
    )
    anim = FuncAnimation(
        fig,
        update,
        iter(region.temperatures_kelvin),
        interval=interval,
        repeat=False,
        cache_frame_data=False,
    )
    return anim
