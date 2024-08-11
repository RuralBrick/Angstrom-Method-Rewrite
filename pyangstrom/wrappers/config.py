import logging
import warnings
from typing import Optional
from copy import copy
import json
from collections import defaultdict

from pyangstrom.config import Config
from pyangstrom.exp_setup import ExperimentalSetup
from pyangstrom.transform import (
    CartesianGeometry,
    PolarGeometry,
    Point,
    RegionStructure,
    RegionBatchConfig,
)
from pyangstrom.fit import SolverInformation, FitterInformation


logger = logging.getLogger('config')


def create_config() -> Config:
    setup: ExperimentalSetup = {
        'meters_per_pixel': 25e-6,
        'material_properties': {},
    }
    config: Config = {
        'experimental_setup': setup,
        'region_information': [],
    }
    return config

def record_to_cartesian_geometry(record) -> Optional[CartesianGeometry]:
    try:
        match record['direction']:
            case 'right-left':
                geometry: CartesianGeometry = {
                    'min_x_pixels': int(record['x_region_line_center']) - int(record['dx']),
                    'max_x_pixels': int(record['x_region_line_center']),
                    'min_y_pixels': int(record['y_region_line_center']) - int(record['dy'])//2,
                    'max_y_pixels': int(record['y_region_line_center']) + int(record['dy'])//2,
                    'heat_source_x_pixels': int(record['x_heater']),
                }
                return geometry
            case 'left-right':
                geometry: CartesianGeometry = {
                    'min_x_pixels': int(record['x_region_line_center']),
                    'max_x_pixels': int(record['x_region_line_center']) + int(record['dx']),
                    'min_y_pixels': int(record['y_region_line_center']) - int(record['dy'])//2,
                    'max_y_pixels': int(record['y_region_line_center']) + int(record['dy'])//2,
                    'heat_source_x_pixels': int(record['x_heater']),
                }
                return geometry
            case 'bottom-up':
                geometry: CartesianGeometry = {
                    'min_x_pixels': int(record['x_region_line_center']) - int(record['dx'])//2,
                    'max_x_pixels': int(record['x_region_line_center']) + int(record['dx'])//2,
                    'min_y_pixels': int(record['y_region_line_center']) - int(record['dy']),
                    'max_y_pixels': int(record['y_region_line_center']),
                    'heat_source_y_pixels': int(record['y_heater']),
                }
                return geometry
            case 'up-bottom':
                geometry: CartesianGeometry = {
                    'min_x_pixels': int(record['x_region_line_center']) - int(record['dx'])//2,
                    'max_x_pixels': int(record['x_region_line_center']) + int(record['dx'])//2,
                    'min_y_pixels': int(record['y_region_line_center']),
                    'max_y_pixels': int(record['y_region_line_center']) + int(record['dy']),
                    'heat_source_y_pixels': int(record['y_heater']),
                }
                return geometry
            case _:
                warnings.warn(
                    f"Direction not recognized: {record['direction']}"
                )
                return None
    except KeyError as e:
        warnings.warn(f"Could not find {e} field for cartesian geometry")
        return None

def record_to_polar_geometries(record) -> Optional[list[PolarGeometry]]:
    try:
        center: Point = {
            'x_pixels': float(record['x0_pixels']),
            'y_pixels': float(record['y0_pixels']),
        }
        geometries = []
        for min_theta, max_theta in json.loads(record['anguler_range']):
            geometry: PolarGeometry = {
                'center': copy(center),
                'min_r_pixels': float(record['R0_pixels']),
                'max_r_pixels': float(record['R0_pixels']) + float(record['R_analysis_pixels']),
                'num_r': int(record['R_analysis_pixels']),
                'min_theta_degrees': float(min_theta),
                'max_theta_degrees': float(max_theta),
                'num_theta': 180,
            }
            geometries.append(geometry)
        return geometries
    except KeyError as e:
        warnings.warn(f"Could not find {e} field for polar geometry")
        return None

def record_to_lopez_baeza_short(record) -> Optional[SolverInformation]:
    try:
        solver: SolverInformation = {
            'name': 'log_lopez-baeza',
            'guesses': {
                'thermal_diffusivity_log10_m2__s': -5,
                'heat_transfer_coefficient_log10_W__m2_K': -2,
            },
            'parameters': {
                'r_meters': float(record['r']),
                'length_meters': float(record['L']),
            },
        }
        return solver
    except KeyError as e:
        warnings.warn(f"Could not find {e} field for lopez_baeza_short solver")
        return None

def record_to_kil_circular_room_temp(record) -> Optional[SolverInformation]:
    try:
        solver: SolverInformation = {
            'name': 'kil',
            'guesses': {
                'thermal_diffusivity_m2__s': 1,
                'convective_heat_transfer_term': 35,
            },
            'parameters': {
                'heating_source_radius_meters': record['R0_pixels'] * 25e-6,
                'outer_boundary_radius_meters': 4.8e-2,
            },
        }
        return solver
    except KeyError as e:
        warnings.warn(f"Could not find {e} field for kil_circular_room_temp solver")
        return None

def record_to_lsr(record) -> Optional[FitterInformation]:
    try:
        fitter: FitterInformation = {
            'name': 'nelder-mead',
        }
        return fitter
    except KeyError as e:
        warnings.warn(f"Could not find {e} field for lsr fitter")
        return None

def exp_condition_to_config(exp_condition: list[dict]) -> dict[str, Config]:
    """Output: dict[recording_name] -> config"""
    dd_config = defaultdict(create_config)
    for record in exp_condition:
        if 'rec_name' in record:
            config: Config = dd_config[record['rec_name']]
        elif 'rec_num' in record:
            config: Config = dd_config[record['rec_num']]
        else:
            warnings.warn(
                "No recording identification field found in exp_condition"
            )
            continue
        geometries = None
        structure: RegionStructure = {'average_out_span': True}
        for key, value in record.items():
            match key:
                case 'rec_name' | 'rec_num':
                    continue
                case 'f_heating':
                    config['experimental_setup']['heating_frequency_hertz'] = float(value)
                case 'cp':
                    config['experimental_setup']['material_properties']['specific_heat_capacity_J__kg_K'] = float(value)
                case 'rho':
                    config['experimental_setup']['material_properties']['density_kg__m3'] = float(value)
                case 'direction' | 'x_region_line_center' | 'y_region_line_center' | 'dx' | 'dy' | 'x_heater' | 'y_heater':
                    if not geometries:
                        geometries = [record_to_cartesian_geometry(record)]
                case 'anguler_range' | 'x0_pixels' | 'y0_pixels' | 'R0_pixels' | 'R_analysis_pixels':
                    if not geometries:
                        geometries = record_to_polar_geometries(record)
                    if 'solver' not in config:
                        config['solver'] = record_to_kil_circular_room_temp(record)
                case 'Nr_pixels':
                    continue
                case 'gap' | 'gap_pixels':
                    structure['num_deinterleaving_groups'] = int(value)
                case 'exp_amp_phase_extraction_method' | 'analysis_method':
                    if 'signal_processor' not in config:
                        config['signal_processor'] = {
                            'name': value,
                            'apply_filter': False,
                        }
                case 'L' | 'r':
                    if 'solver' not in config:
                        config['solver'] = record_to_lopez_baeza_short(record)
                case _:
                    logger.warn(f"Unrecognized field: {key} -> {value}")
        batch_config: RegionBatchConfig = {
            'geometries': geometries,
            'structure': structure,
            'average_over_regions': True,
        }
        config['region_information'].append(batch_config)
        # HACK
        config['fitter'] = record_to_lsr(record)
        # end HACK
    return dict(dd_config)
