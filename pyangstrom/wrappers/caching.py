from pathlib import Path
from dataclasses import dataclass
from typing import get_type_hints
import multiprocessing

import pandas as pd
import numpy as np
from pyangstromHT.high_T_angstrom_method import (
    parallel_temperature_average_batch_experimental_results,
)
from pyangstromRT.blmcmc import calculate_theoretical_results

from pyangstrom.yuan.signatures import TheorConfig
from pyangstrom.wrappers.data_extraction import McmcConfig, FrameConfig
from pyangstrom.wrappers.data_extraction import mcmc_analysis, get_first_frame
from pyangstrom.wrappers.helpers import ht_df_to_rt_df


class FigCacheConfig(McmcConfig, TheorConfig, FrameConfig):
    pass

class CacheConfig(FigCacheConfig):
    pass

class BatchData:
    @dataclass
    class Record:
        df_temp: pd.DataFrame
        df_ht_amp_phase: pd.DataFrame

    def __init__(
            self,
            working_directory: Path | str,
            parameters_file: str,
            *,
            __df_temp=None,
            __df_ht_amp_phase=None
    ) -> None:
        self._p_wd = Path(working_directory)
        self._fn_config = parameters_file
        self.__lst_df_temp = __df_temp
        self.__lst_df_ht_amp_phase = __df_ht_amp_phase

    def iterdata(self):
        if self.__lst_df_temp is None or self.__lst_df_ht_amp_phase is None:
            self.__lst_df_temp, self.__lst_df_ht_amp_phase = parallel_temperature_average_batch_experimental_results(
                self._fn_config,
                code_directory=f'{self._p_wd}/',
                data_directory=f'{self._p_wd / "temperature data"}/',
                num_cores=multiprocessing.cpu_count(),
            )
        return (
            BatchData.Record(*v) for v
            in zip(self.__lst_df_temp, self.__lst_df_ht_amp_phase)
        )

class FigCache:
    def __init__(
            self,
            working_directory: Path | str,
            dict_config: FigCacheConfig,
            df_rt_amp_phase: pd.DataFrame,
            *,
            __df_mcmc_results=None,
            __arr_theor_amp=None,
            __arr_theor_phase=None,
            __arr_first_frame=None,
            __alpha_fitting=None,
            __h_fitting=None,
    ):
        self._p_wd = Path(working_directory)
        self._dict_config = dict_config
        self._df_rt_amp_phase = df_rt_amp_phase
        self.__df_mcmc_results = __df_mcmc_results
        self.__arr_theor_amp = __arr_theor_amp
        self.__arr_theor_phase = __arr_theor_phase
        self.__arr_first_frame = __arr_first_frame
        self.__alpha_fitting = __alpha_fitting
        self.__h_fitting = __h_fitting

    def get_df_mcmc_results(self):
        if self.__df_mcmc_results is None:
            self.__df_mcmc_results = mcmc_analysis(
                self._p_wd,
                self._dict_config,
                self._df_rt_amp_phase,
            )
        return self.__df_mcmc_results

    def get_arr_theor_amp(self):
        if self.__arr_theor_amp is None:
            self.__set_arr_theor()
        return self.__arr_theor_amp

    def get_arr_theor_phase(self):
        if self.__arr_theor_phase is None:
            self.__set_arr_theor()
        return self.__arr_theor_phase

    def get_arr_first_frame(self):
        if self.__arr_first_frame is None:
            self.__arr_first_frame = get_first_frame(
                self._p_wd,
                self._dict_config,
            )
        return self.__arr_first_frame

    def __set_arr_theor(self):
        self.__arr_theor_amp, self.__arr_theor_phase = calculate_theoretical_results(
            self._dict_config,
            self._dict_config,
            self._df_rt_amp_phase,
            [
                self.__get_alpha_fitting(),
                self.__get_h_fitting(),
            ]
        )

    def __get_alpha_fitting(self):
        if not self.__alpha_fitting:
            df_mcmc_results = self.get_df_mcmc_results()
            self.__alpha_fitting = np.mean(df_mcmc_results['alpha'])
        return self.__alpha_fitting

    def __get_h_fitting(self):
        if not self.__h_fitting:
            df_mcmc_results = self.get_df_mcmc_results()
            self.__h_fitting = np.mean(df_mcmc_results['h'])
        return self.__h_fitting

class Cache:
    def __init__(
            self,
            working_directory: Path | str,
            parameters_file: str,
            px=25e-6, # Length of pixel in meters
            *,
            __datatable: BatchData=None,
            __lst_figcache: list=None
    ) -> None:
        self._p_wd = Path(working_directory)
        self._fn_config = parameters_file
        self._px = px
        self.__datatable = __datatable
        self.__lst_figcache = __lst_figcache

    def iterdata(self):
        return zip(self.__get_datatable().iterdata(), self.__get_lst_fig_cache())

    def __get_datatable(self):
        if not self.__datatable:
            self.__datatable = BatchData(self._p_wd, self._fn_config)
        return self.__datatable

    def __get_lst_fig_cache(self):
        if not self.__lst_figcache:
            df_config = pd.read_csv(
                self._p_wd / 'batch process information' / self._fn_config,
                usecols=get_type_hints(CacheConfig).keys(),
                dtype=get_type_hints(CacheConfig),
            )
            datatable = self.__get_datatable()
            self.__lst_figcache = [
                FigCache(
                    self._p_wd,
                    config._asdict(),
                    ht_df_to_rt_df(batchdata.df_ht_amp_phase, self._px),
                )
                for config, batchdata
                in zip(df_config.itertuples(), datatable.iterdata())
            ]
        return self.__lst_figcache
