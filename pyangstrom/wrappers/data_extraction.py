from pathlib import Path
from typing import TypedDict
from collections.abc import Generator

import pandas as pd
import numpy as np
from pyangstromRT.blmcmc import multi_chain_Metropolis_Hasting

from pyangstrom.wrappers.helpers import (
    format_ht_phase_amp_loc_info,
    iter_frame_path,
)


class McmcConfig(TypedDict):
    """For mcmc_analysis"""
    rec_name: str
    L: float
    r: float
    cp: float
    rho: float
    x0_pixels: int
    y0_pixels: int
    gap_pixels: int
    f_heating: float
    exp_amp_phase_extraction_method: str
    N_sample: int
    N_chains: int

class FrameConfig(TypedDict):
    """For get_first_frame"""
    rec_name: str

def mcmc_analysis(
        working_directory: Path | str,
        dict_config: McmcConfig,
        df_rt_amp_phase: pd.DataFrame,
        params_init=[ # Initial estimates of:
             0,     # log(alpha)
             1,     # log(h)
            -2,     # log(sigma_dA)
            -2,     # log(sigma_dP)
             0.2,   # rho
        ],
        prior_log_mu=[ # Prior beliefs of:
             0,     # log(alpha)
             1,     # log(h)
            -2,     # log(sigma_dA)
            -2,     # log(sigma_dP)
             0.5,   # rho
        ],
        prior_log_sigma=[ # Variance of:
            2,  # log(alpha)
            2,  # log(h)
            2,  # log(sigma_dA)
            2,  # log(sigma_dP)
            2,  # rho
        ],
        transition_sigma=[
            0.01,
            0.01,
            0.02,
            0.02,
            0.02,
        ],
        *,
        force_rerun=False,
) -> pd.DataFrame:
    p_wd = Path(working_directory)
    p_results = p_wd / 'mcmc_results_dump' / f'mcmc_{dict_config["N_sample"]}_{format_ht_phase_amp_loc_info(**dict_config)}'
    if force_rerun or not p_results.is_file():
        multi_chain_Metropolis_Hasting(
            directory_path=f'{p_wd}/',
            df_phase_diff_amp_ratio=df_rt_amp_phase,
            phase_amp_loc_info=format_ht_phase_amp_loc_info(**dict_config),
            params_init=params_init,
            prior_log_mu=prior_log_mu,
            prior_log_sigma=prior_log_sigma,
            transition_sigma=transition_sigma,
            analysis_region=dict_config,
            material_properties=dict_config,
            N_sample=dict_config['N_sample'],
            N_chains=dict_config['N_chains'],
            result_name=None, # unused
        )
    df_mcmc_results = pd.read_csv(p_results, index_col=0)
    return df_mcmc_results

def get_frame(
        p_frame: Path,
) -> np.ndarray:
    return np.genfromtxt(
        p_frame,
        delimiter=',',
        skip_header=6,
    )

def get_first_frame(
        working_directory: Path | str,
        dict_config: FrameConfig,
) -> np.ndarray:
    p_wd = Path(working_directory)
    p_ir = p_wd / 'temperature data' / dict_config['rec_name']
    arr_first_frame = get_frame(next(p_ir.iterdir()))
    return arr_first_frame

def get_all_frames(
        working_directory: Path | str,
        dict_config: FrameConfig,
) -> Generator:
    p_wd = Path(working_directory)
    p_ir = p_wd / 'temperature data' / dict_config['rec_name']
    for p_frame in iter_frame_path(p_ir):
        yield get_frame(p_frame)
