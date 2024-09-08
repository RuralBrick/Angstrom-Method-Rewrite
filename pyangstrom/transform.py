import logging
import warnings
from typing import TypedDict, Iterable
from enum import Enum, auto
from dataclasses import dataclass

import pandas as pd
import numpy as np

from pyangstrom.exp_setup import ExperimentalSetup


logger = logging.getLogger('transform')

TEMPERATURE_OFFSET = {
    'Temperature (C)': 273.15,
}

class Direction(Enum):
    LESSER_X = auto()
    GREATER_X = auto()
    LESSER_Y = auto()
    GREATER_Y = auto()


class Point(TypedDict):
    x_pixels: int | float
    y_pixels: int | float

class CartesianGeometry(TypedDict, total=False):
    min_x_pixels: int
    max_x_pixels: int
    min_y_pixels: int
    max_y_pixels: int
    heat_source_x_pixels: int
    heat_source_y_pixels: int

class PolarGeometry(TypedDict):
    center: Point
    min_r_pixels: int | float
    max_r_pixels: int | float
    num_r: int
    min_theta_degrees: int | float
    max_theta_degrees: int | float
    num_theta: int

Geometry = CartesianGeometry | PolarGeometry

class RegionStructure(TypedDict, total=False):
    subtract_temperatures_by: str
    average_out_span: bool
    num_deinterleaving_groups: int

class RegionConfig(TypedDict, total=False):
    geometry: Geometry
    structure: RegionStructure

class RegionBatchConfig(TypedDict, total=False):
    geometries: list[Geometry]
    structure: RegionStructure
    average_over_regions: bool

RegionInformation = RegionConfig | RegionBatchConfig | list[RegionConfig] | list[RegionBatchConfig]


@dataclass
class Margins:
    seconds_elapsed: np.ndarray
    displacements_meters: np.ndarray

    @classmethod
    def new(
            cls,
            df_recording: pd.DataFrame,
            temperatures: np.ndarray,
            min_displacement_pixels: int | float,
            max_displacement_pixels: int | float,
            setup: ExperimentalSetup,
    ) -> "Margins":
        normalized_timestamps = df_recording.index - df_recording.index.min()
        seconds_elapsed = normalized_timestamps.total_seconds().to_numpy()

        disp = np.linspace(
            min_displacement_pixels,
            max_displacement_pixels,
            temperatures.shape[1],
            dtype=float,
        )
        disp = disp * setup['meters_per_pixel']
        disp = np.stack(temperatures.shape[2] * [disp], axis=1)

        return cls(seconds_elapsed, disp)

@dataclass
class Region:
    """A bounded region of IR temperature camera data after undergoing
    transformations and changes of basis until

    Attributes
    ----------
    timestamps
        The timestamps of the original IR camera frames.
    temperatures_kelvin
        An N-dimensional array of temperatures grouped by time, displacement
        from heating source, and other factors based on its axes.
    margins
        The range of each corresponding axis of temperatures_kelvin. Always
        in (time_span, displacement_range_meters, ...) order.
    """
    timestamps: pd.DatetimeIndex
    temperatures_kelvin: np.ndarray
    margins: Margins # TODO: Update docstring


def convert_temperatures_to_kelvin(
        df_recording: pd.DataFrame,
        arr_temps: np.ndarray,
) -> np.ndarray:
    try:
        offset = TEMPERATURE_OFFSET[df_recording['Units'].unique().item()]
        return arr_temps + offset
    except ValueError:
        raise ValueError("More than one temperature unit found")
    except KeyError:
        msg = (f"Temperature unit \"{df_recording['Units'].unique().item()}\" "
               "not accounted for")
        raise NotImplementedError(msg)

def find_heat_source_direction(geometry: CartesianGeometry) -> Direction:
    """
    Raises
    ------
    KeyError
        Field not found in geometry.
    ValueError
        Invalid geometry.
    """
    if ('heat_source_x_pixels' in geometry
            and 'heat_source_y_pixels' in geometry):
        raise ValueError(
            "Cannot have both heat_source_x_pixels and heat_source_y_pixels in "
            "geometry."
        )
    elif 'heat_source_x_pixels' in geometry:
        if geometry['heat_source_x_pixels'] <= geometry['min_x_pixels']:
            return Direction.LESSER_X
        elif geometry['heat_source_x_pixels'] >= geometry['max_x_pixels']:
            return Direction.GREATER_X
        else:
            raise ValueError(
                "heat_source_x_pixels cannot be between min_x_pixels and "
                "max_x_pixels."
            )
    elif 'heat_source_y_pixels' in geometry:
        if geometry['heat_source_y_pixels'] <= geometry['min_y_pixels']:
            return Direction.LESSER_Y
        elif geometry['heat_source_y_pixels'] >= geometry['max_y_pixels']:
            return Direction.GREATER_Y
        else:
            raise ValueError(
                "heat_source_y_pixels cannot be between min_y_pixels and "
                "max_y_pixels."
            )
    else:
        raise KeyError(
            "Must have at either heat_source_x_pixels or heat_source_y_pixels "
            "in geometry."
        )

def extract_cartesian_region(
        df_recording: pd.DataFrame,
        geometry: CartesianGeometry,
        setup: ExperimentalSetup,
) -> Region:
    """
    Raises
    ------
    KeyError
        Field not found in geometry.
    ValueError
        Invalid geometry.
    """
    temps = np.stack(df_recording['Samples']) # (time, height, width)
    temps = np.moveaxis(temps, [-1, -2], [0, 1]) # (width, height, time)
    temps = temps[
        geometry['min_x_pixels'] : geometry['max_x_pixels'] + 1,
        geometry['min_y_pixels'] : geometry['max_y_pixels'] + 1,
    ]
    temps = np.moveaxis(temps, [0, 1], [1, 2]) # (time, width, height)

    # (time, displacement, span)
    match find_heat_source_direction(geometry):
        case Direction.LESSER_X:
            min_disp_px = geometry['min_x_pixels'] - geometry['heat_source_x_pixels']
            max_disp_px = geometry['max_x_pixels'] - geometry['heat_source_x_pixels']
        case Direction.GREATER_X:
            temps = np.flip(temps, axis=1)
            min_disp_px = geometry['heat_source_x_pixels'] - geometry['max_x_pixels']
            max_disp_px = geometry['heat_source_x_pixels'] - geometry['min_x_pixels']
        case Direction.LESSER_Y:
            temps = np.swapaxes(temps, 1, 2)
            min_disp_px = geometry['min_y_pixels'] - geometry['heat_source_y_pixels']
            max_disp_px = geometry['max_y_pixels'] - geometry['heat_source_y_pixels']
        case Direction.GREATER_Y:
            temps = np.swapaxes(temps, 1, 2)
            temps = np.flip(temps, axis=1)
            min_disp_px = geometry['heat_source_y_pixels'] - geometry['max_y_pixels']
            max_disp_px = geometry['heat_source_y_pixels'] - geometry['min_y_pixels']

    temps_kelvin = convert_temperatures_to_kelvin(df_recording, temps)

    region = Region(
        df_recording.index,
        temps_kelvin,
        Margins.new(df_recording, temps, min_disp_px, max_disp_px, setup),
    )
    return region

def extract_polar_region(
        df_recording: pd.DataFrame,
        geometry: PolarGeometry,
        setup: ExperimentalSetup,
) -> Region:
    temps = np.stack(df_recording['Samples']) # (time, height, width)
    temps = np.moveaxis(temps, [-1, -2], [0, 1]) # (width, height, time)

    r_pixels = np.linspace(
        geometry['min_r_pixels'],
        geometry['max_r_pixels'],
        geometry['num_r'],
    )
    theta_degrees = np.deg2rad(np.linspace(
        geometry['min_theta_degrees'],
        geometry['max_theta_degrees'],
        geometry['num_theta'],
    ))
    r_coord, theta_coord = np.meshgrid(r_pixels, theta_degrees, indexing='ij')

    x_coord = r_coord * np.cos(theta_coord) + geometry['center']['x_pixels']
    y_coord = r_coord * np.sin(theta_coord) + geometry['center']['y_pixels']

    x_coord_floor = np.floor(x_coord).astype('int')
    x_coord_floorp1 = x_coord_floor + 1
    y_coord_floor = np.floor(y_coord).astype('int')
    y_coord_floorp1 = y_coord_floor + 1

    lower_x_weight = (x_coord_floorp1 - x_coord) / (x_coord_floorp1 - x_coord_floor)
    upper_x_weight = (x_coord - x_coord_floor) / (x_coord_floorp1 - x_coord_floor)
    lower_y_weight = (y_coord_floorp1 - y_coord) / (y_coord_floorp1 - y_coord_floor)
    upper_y_weight = (y_coord - y_coord_floor) / (y_coord_floorp1 - y_coord_floor)

    # (time, radius, angle)
    lxly_temps = np.moveaxis(temps[x_coord_floor, y_coord_floor], [0, 1], [1, 2]) * lower_x_weight * lower_y_weight
    lxuy_temps = np.moveaxis(temps[x_coord_floor, y_coord_floorp1], [0, 1], [1, 2]) * lower_x_weight * upper_y_weight
    uxly_temps = np.moveaxis(temps[x_coord_floorp1, y_coord_floor], [0, 1], [1, 2]) * upper_x_weight * lower_y_weight
    uxuy_temps = np.moveaxis(temps[x_coord_floorp1, y_coord_floorp1], [0, 1], [1, 2]) * upper_x_weight * upper_y_weight

    temps_trans = lxly_temps + lxuy_temps + uxly_temps + uxuy_temps
    temps_kelvin = convert_temperatures_to_kelvin(df_recording, temps_trans)

    region = Region(
        df_recording.index,
        temps_kelvin,
        Margins.new(
            df_recording,
            temps_kelvin,
            geometry['min_r_pixels'],
            geometry['max_r_pixels'],
            setup,
        ),
    )
    return region

def geometry_to_region(
        df_recording: pd.DataFrame,
        geometry: Geometry,
        setup: ExperimentalSetup,
) -> Region:
    match geometry:
        case {'min_x_pixels': _}:
            return extract_cartesian_region(df_recording, geometry, setup)
        case {'center': _}:
            return extract_polar_region(df_recording, geometry, setup)

def truncate_region(region: Region, num_truncate: int, axis: int) -> Region:
    new_time = region.timestamps

    new_temps = np.moveaxis(region.temperatures_kelvin, axis, 0)
    new_temps = new_temps[:-num_truncate]
    new_temps = np.moveaxis(new_temps, 0, axis)

    if axis == 0:
        new_time = region.timestamps[:-num_truncate]
        new_elapsed = region.margins.seconds_elapsed[:-num_truncate]
        new_margins = Margins(new_elapsed, region.margins.displacements_meters)
    else:
        new_disp = region.margins.displacements_meters
        new_disp = np.moveaxis(new_disp, axis - 1, 0)
        new_disp = new_disp[:-num_truncate]
        new_disp = np.moveaxis(new_disp, 0, axis - 1)
        new_margins = Margins(region.margins.seconds_elapsed, new_disp)

    new_region = Region(
        new_time,
        new_temps,
        new_margins,
    )
    return new_region

def restructure_region(region: Region, structure: RegionStructure) -> Region:
    if 'subtract_temperatures_by' in structure:
        match structure['subtract_temperatures_by']:
            case 'mean' | 'avg' | 'average':
                subtrahend = region.temperatures_kelvin.mean()
            case 'min' | 'minimum' | 'lowest':
                subtrahend = region.temperatures_kelvin.min()
            case _:
                subtrahend = 0
                warnings.warn(
                    f"Subtract temperatures by "
                    f"\'{structure['subtract_temperatures_by']}\' "
                    f"not understood"
                )
        region = Region(
            region.timestamps,
            region.temperatures_kelvin - subtrahend,
            region.margins,
        )
    if 'average_out_span' in structure and structure['average_out_span']:
        region = Region(
            region.timestamps,
            region.temperatures_kelvin.mean(axis=2),
            Margins(
                region.margins.seconds_elapsed,
                region.margins.displacements_meters.mean(axis=1),
            ),
        )
    if 'num_deinterleaving_groups' in structure:
        num_disp = region.temperatures_kelvin.shape[1]
        new_num_disp, remainder = divmod(
            num_disp,
            structure['num_deinterleaving_groups'],
        )
        if remainder != 0:
            region = truncate_region(region, remainder, axis=1)
        lst_groups = np.split(region.temperatures_kelvin, new_num_disp, axis=1)
        new_temps = np.stack(lst_groups, axis=1)
        lst_disp = np.split(
            region.margins.displacements_meters,
            new_num_disp,
            axis=0,
        )
        new_disp = np.stack(lst_disp, axis=0)
        region = Region(
            region.timestamps,
            new_temps,
            Margins(region.margins.seconds_elapsed, new_disp),
        )
    return region

def all_timestamps_same(regions: Iterable[Region]) -> bool:
    idx0 = next(iter(regions)).timestamps
    for region in regions:
        if not idx0.symmetric_difference(region.timestamps).empty:
            return False
    return True

def all_temps_same_shape(regions: Iterable[Region]) -> bool:
    dim_sizes = zip(*(r.temperatures_kelvin.shape for r in regions))
    size_counts = [set(s) for s in dim_sizes]
    return all(len(c) == 1 for c in size_counts)

def min_temps_shape(regions: Iterable[Region]) -> tuple:
    dim_sizes = zip(*(r.temperatures_kelvin.shape for r in regions))
    min_shape = tuple(min(s) for s in dim_sizes)
    return min_shape

def trim_regions(regions: Iterable[Region]) -> list[Region]:
    min_shape = min_temps_shape(regions)
    new_regions = []
    for region in regions:
        for axis, size in enumerate(min_shape):
            region = truncate_region(
                region,
                region.temperatures_kelvin.shape[axis] - size,
                axis,
            )
            new_regions.append(region)
    return new_regions

def fully_extract_region(
        df_recording: pd.DataFrame,
        information: RegionInformation,
        setup: ExperimentalSetup,
) -> Region | list[Region] | list[list[Region]]:
    """
    Raises
    ------
    ValueError
        Malformed information.
    """
    match information:
        case {'geometry': geometry}:
            region = geometry_to_region(df_recording, geometry, setup)
            if 'structure' in information:
                region = restructure_region(region, information['structure'])
            return region
        case {'geometries': geometries}:
            regions = [
                geometry_to_region(df_recording, g, setup) for g in geometries
            ]
            if 'structure' in information:
                regions = [
                    restructure_region(r, information['structure'])
                    for r in regions
                ]
            if ('average_over_regions' not in information
                or not information['average_over_regions']):
                return regions
            assert all_timestamps_same(regions)
            if not all_temps_same_shape(regions):
                warnings.warn(
                    "Not all regions have the same number of samples. Trimming "
                    "regions to match minimum shape."
                )
                regions = trim_regions(regions)
            new_temps = np.stack(
                [r.temperatures_kelvin for r in regions],
                axis=2,
            )
            new_temps = new_temps.mean(axis=2)
            new_disp = np.stack(
                [r.margins.displacements_meters for r in regions],
                axis=1
            )
            new_disp = new_disp.mean(axis=1)
            new_region = Region(
                regions[0].timestamps,
                new_temps,
                Margins(regions[0].margins.seconds_elapsed, new_disp),
            )
            return new_region
        case [*region_configs]:
            regions = [
                fully_extract_region(df_recording, c, setup)
                for c in region_configs
            ]
            return regions
        case _:
            raise ValueError(f"Invalid information format: {information}")

def collapse_region(region: Region) -> Region:
    num_times, num_disp, *_ = region.temperatures_kelvin.shape
    new_temps = (region.temperatures_kelvin
                       .reshape(num_times, num_disp, -1)
                       .mean(axis=2))
    new_disp = (region.margins.displacements_meters
                      .reshape(num_disp, -1)
                      .mean(axis=1))
    new_region = Region(
        region.timestamps,
        new_temps,
        Margins(region.margins.seconds_elapsed, new_disp),
    )
    return new_region
