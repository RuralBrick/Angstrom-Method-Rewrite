from typing import TypedDict, NamedTuple
from dataclasses import dataclass
from collections import namedtuple

import pandas as pd
import numpy as np

from pyangstrom.exp_setup import ExperimentalSetup


TEMPERATURE_OFFSET = {
    'Temperature (C)': 273.15,
}

class Direction:
    LESSER_X = {'lesser_x', 'negative_x', 'neg_x'}
    GREATER_X = {'greater_x', 'positive_x', 'pos_x'}
    LESSER_Y = {'lesser_y', 'negative_y', 'neg_y'}
    GREATER_Y = {'greater_y', 'positive_y', 'pos_y'}


class Point(TypedDict):
    x_pixels: int | float
    y_pixels: int | float

class CartesianGeometry(TypedDict):
    min_x_pixels: int
    max_x_pixels: int
    min_y_pixels: int
    max_y_pixels: int
    heat_source: str

class PolarGeometry(TypedDict):
    center: Point
    min_r_pixels: int | float
    max_r_pixels: int | float
    num_r: int
    min_theta_degrees: int | float
    max_theta_degrees: int | float
    num_theta: int

Geometry = CartesianGeometry | PolarGeometry

class RegionStructure(TypedDict):
    average_out_span: bool
    num_deinterleaving_groups: int

class RegionConfig(TypedDict):
    geometry: Geometry
    structure: RegionStructure

class RegionBatchConfig(TypedDict):
    average_over_regions: bool


Margins = NamedTuple

@dataclass
class Region:
    time: pd.DatetimeIndex
    temps_kelvin: np.ndarray
    margins: Margins # Should be in (time, displacement_meters, ...) order


def convert_temps_to_kelvin(
        df_recording: pd.DataFrame,
        arr_temps: np.ndarray,
) -> np.ndarray:
    try:
        return arr_temps + TEMPERATURE_OFFSET[df_recording['Units'].unique().item()]
    except ValueError:
        raise ValueError("More than one temperature unit found")
    except KeyError:
        msg = (f"Temperature unit \"{df_recording['Units'].unique().item()}\" "
               "not accounted for")
        raise NotImplementedError(msg)

def extract_cartesian_region(
        df_recording: pd.DataFrame,
        geometry: CartesianGeometry,
        setup: ExperimentalSetup,
) -> Region:
    temps = np.stack(df_recording['Samples']) # (time, height, width)
    temps = np.moveaxis(temps, [-1, -2], [0, 1]) # (width, height, time)
    temps = temps[
        geometry['min_x_pixels'] : geometry['max_x_pixels'] + 1,
        geometry['min_y_pixels'] : geometry['max_y_pixels'] + 1,
    ]
    temps = np.moveaxis(temps, [0, 1], [1, 2]) # (time, width, height)

    # (time, displacement, span)
    if geometry['heat_source'] in Direction.LESSER_X:
        margins = namedtuple(
            'XAlignedCartesianMargins',
            'time_span x_range_meters y_range_meters',
        )(
            df_recording.index.max() - df_recording.index.min(),
            (geometry['max_x_pixels'] - geometry['min_x_pixels']) * setup['meters_per_pixel'],
            (geometry['max_y_pixels'] - geometry['min_y_pixels']) * setup['meters_per_pixel'],
        )
    elif geometry['heat_source'] in Direction.GREATER_X:
        temps = np.flip(temps, axis=1)
        margins = namedtuple(
            'XAlignedCartesianMargins',
            'time_span x_range_meters y_range_meters',
        )(
            df_recording.index.max() - df_recording.index.min(),
            (geometry['max_x_pixels'] - geometry['min_x_pixels']) * setup['meters_per_pixel'],
            (geometry['max_y_pixels'] - geometry['min_y_pixels']) * setup['meters_per_pixel'],
        )
    elif geometry['heat_source'] in Direction.LESSER_Y:
        temps = np.swapaxes(temps, 1, 2)
        margins = namedtuple(
            'YAlignedCartesianMargins',
            'time_span y_range_meters x_range_meters',
        )(
            df_recording.index.max() - df_recording.index.min(),
            (geometry['max_y_pixels'] - geometry['min_y_pixels']) * setup['meters_per_pixel'],
            (geometry['max_x_pixels'] - geometry['min_x_pixels']) * setup['meters_per_pixel'],
        )
    elif geometry['heat_source'] in Direction.GREATER_Y:
        temps = np.swapaxes(temps, 1, 2)
        temps = np.flip(temps, axis=1)
        margins = namedtuple(
            'YAlignedCartesianMargins',
            'time_span y_range_meters x_range_meters',
        )(
            df_recording.index.max() - df_recording.index.min(),
            (geometry['max_y_pixels'] - geometry['min_y_pixels']) * setup['meters_per_pixel'],
            (geometry['max_x_pixels'] - geometry['min_x_pixels']) * setup['meters_per_pixel'],
        )

    temps_kelvin = convert_temps_to_kelvin(df_recording, temps)

    region = Region(
        df_recording.index,
        temps_kelvin,
        margins,
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
    temps_kelvin = convert_temps_to_kelvin(df_recording, temps_trans)

    margins = namedtuple('PolarMargins', 'time_span r_range_meters theta_range_degrees')(
        df_recording.index.max() - df_recording.index.min(),
        (geometry['max_r_pixels'] - geometry['min_r_pixels']) * setup['meters_per_pixel'],
        geometry['max_theta_degrees'] - geometry['min_theta_degrees'],
    )

    region = Region(
        df_recording.index,
        temps_kelvin,
        margins,
    )
    return region

def collapse_region(region: Region) -> Region:
    time_span, disp_range, *_ = region.margins
    new_margins = namedtuple('BaseMargins', 'time_span displacement_range_meters')(
        time_span,
        disp_range,
    )
    num_times, num_disp, *_ = region.temps_kelvin.shape
    new_region = Region(
        region.time,
        region.temps_kelvin.reshape(num_times, num_disp, -1).mean(axis=2),
        new_margins,
    )
    return new_region

def restructure_region(region: Region, structure: RegionStructure) -> Region:
    pass
