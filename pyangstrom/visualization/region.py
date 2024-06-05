import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Wedge, Circle
from matplotlib.animation import Animation, FuncAnimation

from pyangstrom.transform import (
    find_heat_source_direction,
    Direction,
    CartesianGeometry,
    PolarGeometry,
    Geometry,
    Region,
    collapse_region,
)
from pyangstrom.visualization.recording import plot_recording


def add_cartesian_geometry(
        ax: Axes,
        geometry: CartesianGeometry,
) -> Axes:
    ax.add_patch(Rectangle(
        (geometry['min_x_pixels'], geometry['min_y_pixels']),
        geometry['max_x_pixels'] - geometry['min_x_pixels'],
        geometry['max_y_pixels'] - geometry['min_y_pixels'],
        hatch='..',
        edgecolor='red',
        facecolor='none',
    ))
    match find_heat_source_direction(geometry):
        case Direction.LESSER_X:
            ax.plot(
                [geometry['heat_source_x_pixels'], geometry['heat_source_x_pixels']],
                [geometry['min_y_pixels'], geometry['max_y_pixels']],
                linewidth=2,
                color='blue',
            )
        case Direction.GREATER_X:
            ax.plot(
                [geometry['heat_source_x_pixels'], geometry['heat_source_x_pixels']],
                [geometry['min_y_pixels'], geometry['max_y_pixels']],
                linewidth=2,
                color='blue',
            )
        case Direction.LESSER_Y:
            ax.plot(
                [geometry['min_x_pixels'], geometry['max_x_pixels']],
                [geometry['heat_source_y_pixels'], geometry['heat_source_y_pixels']],
                linewidth=2,
                color='blue',
            )
        case Direction.GREATER_Y:
            ax.plot(
                [geometry['min_x_pixels'], geometry['max_x_pixels']],
                [geometry['heat_source_y_pixels'], geometry['heat_source_y_pixels']],
                linewidth=2,
                color='blue',
            )
    return ax

def add_polar_geometry(
        ax: Axes,
        geometry: PolarGeometry,
) -> Axes:
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

def add_geometry(
        ax: Axes,
        geometry: Geometry | list[Geometry],
) -> Axes:
    if isinstance(geometry, list):
        for g in geometry:
            ax = add_geometry(ax, g)
        return ax
    match geometry:
        case {'min_x_pixels': _}:
            return add_cartesian_geometry(ax, geometry)
        case {'center': _}:
            return add_polar_geometry(ax, geometry)

def plot_geometry(
        ax: Axes,
        df_recording: pd.DataFrame,
        geometry: Geometry | list[Geometry],
) -> Axes:
    ax = plot_recording(ax, df_recording)
    ax = add_geometry(ax, geometry)
    return ax

def plot_spatiotemporal_heat_map(
        ax: Axes,
        region: Region,
) -> Axes:
    region = collapse_region(region)
    ax.imshow(region.temperatures_kelvin)
    ax.set_title("Spatiotemporal Heat Map of Analysis Region")
    ax.set_xlabel("Displacement from heat source (meters)")
    ax.set_ylabel("Frames elapsed")
    return ax

def idx_displacement_to_label(idx_displacement: int):
    idx_displacement += 1
    if idx_displacement == 0:
        return 'Line N'
    elif idx_displacement < 0:
        return f'Line N{idx_displacement}'
    else:
        return f'Line {idx_displacement}'

def plot_isotherms(
        ax: Axes,
        region: Region,
        idx_displacements: list[int] = [0, -1],
        use_timestamps: bool = False,
) -> Axes:
    region = collapse_region(region)
    if use_timestamps:
        x = region.timestamps
        ax.set_xlabel("Time")
    else:
        x = region.margins.seconds_elapsed
        ax.set_xlabel("Time elapsed (seconds)")
    labels = map(idx_displacement_to_label, idx_displacements)
    for idx, label in zip(idx_displacements, labels):
        ax.plot(x, region.temperatures_kelvin[:, idx], label=label)
    ax.set_ylabel("Temperature (kelvin)")
    ax.set_title("Temperatures of Isotherms Over Time")
    ax.legend()
    return ax

def plot_groups(ax: Axes, region: Region, use_timestamps: bool = False) -> Axes:
    if use_timestamps:
        x = region.timestamps
        ax.set_xlabel("Ttime")
    else:
        x = region.margins.seconds_elapsed
        ax.set_xlabel("Time elapsed (seconds)")
    ax.plot(x, region.temperatures_kelvin.mean(axis=1), alpha=0.5)
    ax.set_ylabel("Temperature (kelvin)")
    ax.set_title("Average Temperatures of Grouped Nodes Over Time")
    return ax

def animate_region(region: Region) -> Animation:
    ax: Axes = None
    fig, ax = plt.subplots()
    region = collapse_region(region)
    ln, = ax.plot(
        region.margins.displacements_meters,
        region.temperatures_kelvin[0],
    )
    ax.set_ylim(
        region.temperatures_kelvin.min(),
        region.temperatures_kelvin.max(),
    )
    ax.set_title(region.timestamps[0].strftime('(%b %d) %H:%M:%S.%f'))
    ax.set_xlabel("Displacement from heat source (meters)")
    ax.set_ylabel("Temperature (kelvin)")
    def update(frame):
        time, temps = frame
        ln.set_data(region.margins.displacements_meters, temps)
        ax.set_title(time.strftime('(%b %d) %H:%M:%S.%f'))
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
        iter(zip(region.timestamps, region.temperatures_kelvin)),
        interval=interval,
        repeat=False,
        cache_frame_data=False,
    )
    return anim
