from typing import NamedTuple

import pandas as pd
import numpy as np


TEMPERATURE_OFFSET = {
    'Temperature (C)': 273.15,
}

class Region:
    def __init__(self) -> None:
        """Always in absolute temperature (expect Kelvin)"""
        self.margins: NamedTuple = None # Should be in (dist, ..., time) order

class Geometry:
    pass

class PolarGeometry(Geometry):
    pass


def add_temperature_offset(
        df_recording: pd.DataFrame,
        arr_temps: np.ndarray
) -> np.ndarray:
    try:
        return arr_temps + TEMPERATURE_OFFSET[df_recording['Units'].unique().item()]
    except ValueError:
        raise ValueError("More than one temperature unit found")
    except KeyError:
        msg = (f"Temperature unit \"{df_recording['Units'].unique().item()}\" "
               "not accounted for")
        raise NotImplementedError(msg)

def extract_polar_region(
        df_recording: pd.DataFrame,
        geometry: PolarGeometry,
) -> Region:
    temps = np.stack(df_recording['Samples']) # (time, ..., height, width)
    temps = np.moveaxis(temps, [0, -1, -2], [-1, 0, 1]) # (width, height, ..., time)

    r = np.linspace(
        geometry.min_r,
        geometry.max_r,
        geometry.num_r,
    )
    theta = np.deg2rad(np.linspace(
        geometry.min_theta,
        geometry.max_theta,
        geometry.num_theta,
    ))
    r_coord, theta_coord = np.meshgrid(r, theta)

    x_coord = r_coord * np.cos(theta_coord) + geometry.center.x
    y_coord = r_coord * np.sin(theta_coord) + geometry.center.y

    x_coord_floor = np.floor(x_coord).astype('int')
    x_coord_floorp1 = x_coord_floor + 1
    y_coord_floor = np.floor(y_coord).astype('int')
    y_coord_floorp1 = y_coord_floor + 1

    lower_x_weight = (x_coord_floorp1 - x_coord) / (x_coord_floorp1 - x_coord_floor)
    upper_x_weight = (x_coord - x_coord_floor) / (x_coord_floorp1 - x_coord_floor)
    lower_y_weight = (y_coord_floorp1 - y_coord) / (y_coord_floorp1 - y_coord_floor)
    upper_y_weight = (y_coord - y_coord_floor) / (y_coord_floorp1 - y_coord_floor)

    # (..., time, radius, angle)
    lxly_temps = np.moveaxis(temps[x_coord_floor, y_coord_floor], [0, 1], [-2, -1]) * lower_x_weight * lower_y_weight
    lxuy_temps = np.moveaxis(temps[x_coord_floor, y_coord_floorp1], [0, 1], [-2, -1]) * lower_x_weight * upper_y_weight
    uxly_temps = np.moveaxis(temps[x_coord_floorp1, y_coord_floor], [0, 1], [-2, -1]) * upper_x_weight * lower_y_weight
    uxuy_temps = np.moveaxis(temps[x_coord_floorp1, y_coord_floorp1], [0, 1], [-2, -1]) * upper_x_weight * upper_y_weight

    temps_trans = lxly_temps + lxuy_temps + uxly_temps + uxuy_temps
    temps_trans = np.moveaxis(temps_trans, [-2, -1], [0, 1]) # (radius, angle, ..., time)
    temps_trans = add_temperature_offset(df_recording, temps_trans)

    region = Region(
        df_recording.index,
        temps_trans,
        (
            geometry.max_r - geometry.min_r,
            geometry.max_theta - geometry.min_theta,
            ...,
            df_recording.index.max() - df_recording.index.min(),
        ),
    )
    return region
