def format_rt_phase_amp_loc_info(
        file_name,
        f_heating,
        x_region_line_center,
        y_region_line_center,
        dx,
        dy,
        x_heater,
        y_heater,
        gap,
        direction,
        **_,
):
    return (f'{file_name}_freq_{1000*f_heating:.0f}'
            f'x{x_region_line_center}y{y_region_line_center}_'
            f'dx{dx}dy{dy}xh{x_heater}yh{y_heater}gap{gap}{direction}.csv')

def format_ht_phase_amp_loc_info(
        rec_name,
        f_heating,
        x0_pixels,
        y0_pixels,
        R0_pixels,
        R_analysis_pixels,
        anguler_range,
        gap_pixels,
        exp_amp_phase_extraction_method,
        **_,
):
    return (f'{rec_name}_freq_{1000*f_heating:.0f}'
            f'x{x0_pixels}y{y0_pixels}_R0{R0_pixels}RA{R_analysis_pixels}'
            f'ar{anguler_range}gap{gap_pixels}'
            f'{exp_amp_phase_extraction_method}.csv')

def ht_df_to_rt_df(df, px):
    df = df.rename(columns={'r_pixels': 'x', 'r_ref_pixels': 'x_ref'})
    df['x'] *= px
    df['x_ref'] *= px
    return df

def iter_frame_path(p_data):
    return sorted(p_data.iterdir(), key=lambda p: int(p.stem.split('_')[-1]))
