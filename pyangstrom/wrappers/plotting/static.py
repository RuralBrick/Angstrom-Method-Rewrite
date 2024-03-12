from typing import TypedDict, NamedTuple, get_type_hints
from pathlib import Path
from functools import partial
from operator import itemgetter

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from pyangstrom.yuan.HT import HtAmpPhaseConfig
from pyangstrom.wrappers.caching import (
    FigCacheConfig,
    FigCache,
    Cache,
    BatchData,
)
from pyangstrom.wrappers.helpers import ht_df_to_rt_df


class SectorConfig(TypedDict):
    x0_pixels: int
    y0_pixels: int
    R0_pixels: int
    R_analysis_pixels: int
    anguler_range: str

class FigureConfig(FigCacheConfig, SectorConfig):
    pass

class Config(HtAmpPhaseConfig, FigureConfig):
    pass

class FigurePlottingResults(NamedTuple):
    figure: plt.Figure
    figcache: FigCache

class AllPlottingResults(NamedTuple):
    lst_figures: list
    cache: Cache

def plot_alpha_hist(ax, label_font_size, df_mcmc_results):
    ax.hist(1e4 * 10**df_mcmc_results['alpha'], bins=20)
    ax.set_xlabel(r'${\alpha}$ (cm$^2$/s)', fontsize=label_font_size, fontweight='bold')
    return ax

def plot_h_hist(ax, label_font_size, df_mcmc_results):
    ax.hist(10**df_mcmc_results['h'], bins=20)
    ax.set_xlabel('$h$ (W/m2K)', fontsize=label_font_size, fontweight='bold')
    return ax

def plot_amp(ax, label_font_size, df_rt_amp_phase, arr_theor_amp):
    ax.plot(1e3 * df_rt_amp_phase['x'], df_rt_amp_phase['amp_ratio'], label='measurement', alpha = 0.4, marker = 'o', linewidth = 0)
    ax.plot(1e3 * df_rt_amp_phase['x'], arr_theor_amp, label='fitting', color='red',marker = 'o', linewidth = 0)
    ax.set_xlabel('x(mm)', fontsize=label_font_size, fontweight='bold')
    ax.set_ylabel('Amplitude decay', fontsize=label_font_size, fontweight='bold')
    ax.legend(prop={'weight': 'bold', 'size': 12})
    return ax

def plot_phase(ax, label_font_size, df_rt_amp_phase, arr_theor_phase):
    ax.plot(1e3 * df_rt_amp_phase['x'], df_rt_amp_phase['phase_diff'], label='measurement', alpha = 0.4, marker = 'o', linewidth = 0)
    ax.plot(1e3 * df_rt_amp_phase['x'], arr_theor_phase, label='fitting', color='red',marker = 'o', linewidth = 0)
    ax.set_xlabel('x(mm)', fontsize=label_font_size, fontweight='bold')
    ax.set_ylabel('Phase difference', fontsize=label_font_size, fontweight='bold')
    ax.legend(prop={'weight': 'bold', 'size': 12})
    return ax

def plot_temp_wave(ax, label_font_size, df_temp):
    idx_line = [c for c in df_temp.columns if isinstance(c, int)]
    ax.plot(df_temp['reltime'], df_temp[min(idx_line)], label='line 0')
    ax.plot(df_temp['reltime'], df_temp[max(idx_line)], label='line N')
    ax.set_xlabel('Time (s)', fontsize=label_font_size, fontweight='bold')
    ax.set_ylabel('Temperature (C)', fontsize=label_font_size, fontweight='bold')
    ax.legend(prop={'weight': 'bold', 'size': 12})
    return ax

def plot_sectors(ax, label_font_size, dict_config: SectorConfig, arr_first_frame):
    ax.imshow(arr_first_frame)
    x0, y0, R0, R_analysis, anguler_range = itemgetter('x0_pixels', 'y0_pixels', 'R0_pixels', 'R_analysis_pixels', 'anguler_range')(dict_config)
    for theta1, theta2 in eval(anguler_range):
        ax.add_patch(Wedge((x0, y0), R0 + R_analysis, theta1, theta2, hatch='..', edgecolor='red', facecolor='none'))
    ax.add_patch(plt.Circle((x0, y0), R0, linewidth=2, edgecolor='r', linestyle='solid', facecolor='none'))
    ax.add_patch(plt.Circle((x0, y0), R0 + R_analysis, linewidth=2, edgecolor='r', linestyle='solid', facecolor='none'))
    return ax

def plot_one_case_figure(
        working_directory: Path | str,
        dict_config: FigureConfig,
        batchdata: BatchData.Record,
        px=25e-6, # Length of pixel in meters
        label_font_size=18,
        *,
        _figcache: FigCache=None,
) -> FigurePlottingResults:
    p_wd = Path(working_directory)
    if not _figcache:
        _figcache = FigCache(
            p_wd,
            dict_config,
            ht_df_to_rt_df(batchdata.df_ht_amp_phase, px),
        )
    plots = [
        [
            # TODO: mcmc[chain_num] --> line plot
            # mcmc[alpha/h] --> hist
            partial(
                plot_alpha_hist,
                df_mcmc_results=_figcache.get_df_mcmc_results()
            ),
            partial(
                plot_h_hist,
                df_mcmc_results=_figcache.get_df_mcmc_results()
            ),
            # TODO: mcmc[alpha/h] -|> auto_correlation --> line plot
            # temp --> line plot
            partial(plot_temp_wave, df_temp=batchdata.df_temp),
            # TODO: temp -|> fit -|> abs --> line plot
            # TODO: temp -|> fit -|> angle --> line plot
        ],
        [
            # ap --> scatter plot
            partial(
                plot_amp,
                df_rt_amp_phase=ht_df_to_rt_df(batchdata.df_ht_amp_phase, px),
                arr_theor_amp=_figcache.get_arr_theor_amp(),
            ),
            partial(
                plot_phase,
                df_rt_amp_phase=ht_df_to_rt_df(batchdata.df_ht_amp_phase, px),
                arr_theor_phase=_figcache.get_arr_theor_phase(),
            ),
            # TODO: properties --> "ap" --> scatter plot
            # IR --> image
            partial(
                plot_sectors,
                dict_config=dict_config,
                arr_first_frame=_figcache.get_arr_first_frame(),
            ),
        ],
    ]
    assert len(set(map(len, plots))) == 1
    fig, axes = plt.subplots(
        len(plots),
        len(plots[0]),
        figsize=(8*len(plots[0]), 8*len(plots))
    )
    for lst_plot, lst_ax in zip(plots, axes):
        for plot, ax in zip(lst_plot, lst_ax):
            plot(ax, label_font_size)
            for t in [*ax.xaxis.get_major_ticks(), *ax.yaxis.get_major_ticks()]:
                t.label.set_fontsize(fontsize=12)
                t.label.set_fontweight('bold')
    return fig, _figcache

def plot_figures_from_parameters(
        working_directory: Path | str,
        parameters_file: str,
        px=25e-6, # Length of pixel in meters
        *,
        _cache: Cache=None,
) -> AllPlottingResults:
    p_wd = Path(working_directory)
    if not _cache:
        _cache = Cache(p_wd, parameters_file, px)
    df_config = pd.read_csv(
        p_wd / 'batch process information' / parameters_file,
        usecols=get_type_hints(FigureConfig).keys(),
        dtype=get_type_hints(FigureConfig),
    )
    gen_fig_results = (
        plot_one_case_figure(p_wd, config._asdict(), batchdata, px, _figcache=figcache)
        for config, (batchdata, figcache)
        in zip(df_config.itertuples(), _cache.iterdata())
    )
    lst_fig, _ = list(zip(*gen_fig_results))
    return lst_fig, _cache

def main():
    print(f"{get_type_hints(Config)=}")

if __name__ == '__main__':
    main()
