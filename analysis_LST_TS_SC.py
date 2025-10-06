# Import necessary libraries
import pandas as pd
import xarray as xr
import os
import sys
import numpy as np
import re
import seaborn as sns 
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error as rmse
from scipy.stats import pearsonr
import scienceplots
import matplotlib.dates as mdates

# local utils (unused here but kept as in your codebase)
sys.path.append('/home/khanalp/code/PhD/utils')
from utils import read_csv_file_with_station_name
from unit_conversion import convert_umolCO2_to_kgC

# ---------------- Plot style (unchanged) ----------------
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use(['science', 'no-latex'])
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.titleweight' : 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,   # embed TrueType in PDF
    'ps.fonttype': 42,    # embed TrueType in PS
})

# ---------------- Helper functions (unchanged logic) ----------------
def temperature_from_lw_out(lw_out: np.ndarray, lw_in:np.ndarray, emissivity: float) -> np.ndarray:
    """Return temperature (K) from LW_OUT/LW_IN using Stefan–Boltzmann law."""
    sigma = 5.670374419e-8  # W/m²/K⁴
    if not (0 < emissivity <= 1):
        raise ValueError("Emissivity must be in the range (0, 1].")
    lw_out = np.asarray(lw_out); lw_in = np.asarray(lw_in)
    if np.any(lw_out < 0):
        raise ValueError("LW_OUT values must be non-negative.")
    return ((lw_out - ((1-emissivity) * lw_in)) / (emissivity * sigma)) ** 0.25

def compute_lst_from_model(df_leaftemp: pd.DataFrame, df_surftemp: pd.DataFrame, df_gap: pd.DataFrame) -> pd.Series:
    """Weighted sunlit/shaded mean (31 elements per step) -> LST (same units as inputs)."""
    sunlit_leaf_temp = df_leaftemp.iloc[:, 0:30]
    shaded_leaf_temp = df_leaftemp.iloc[:, 30:60]
    shaded_soil_temp = df_surftemp["Ts(1)"]
    sunlit_soil_temp = df_surftemp["Ts(2)"]
    Ps = df_gap.iloc[:, 0:31]
    sunlit_all = pd.concat([sunlit_leaf_temp, sunlit_soil_temp], axis=1)
    shaded_all = pd.concat([shaded_leaf_temp, shaded_soil_temp], axis=1)
    lst = (Ps * sunlit_all + (1 - Ps) * shaded_all).mean(axis=1)
    return lst

def align_four(
    df_ref: pd.DataFrame,
    ref_col: str,
    df_model_vBL: pd.DataFrame,
    df_model_vMLWV: pd.DataFrame,
    df_model_vMLHF: pd.DataFrame,
    model_col: str = "LST",
    tolerance: str = "30min",
    ref_label: str = "REF",
) -> pd.DataFrame:
    """
    Align vBL, vMLWV, and vMLHF to reference times (nearest within tolerance),
    then keep only rows where all three models matched.

    Returns
    -------
    pd.DataFrame
        Index: reference timestamps
        Columns: [ref_label, 'vBL', 'vMLWV', 'vMLHF']
    """
    # Reference
    ref = df_ref[[ref_col]].dropna().copy()
    ref.index = pd.to_datetime(ref.index)
    ref = ref.sort_index()
    ref_t = ref.assign(t_ref=ref.index).reset_index(drop=True)

    # Models
    bl = df_model_vBL[[model_col]].dropna().copy()
    mlwv = df_model_vMLWV[[model_col]].dropna().copy()
    mlhf = df_model_vMLHF[[model_col]].dropna().copy()

    for d in (bl, mlwv, mlhf):
        d.index = pd.to_datetime(d.index)
        d.sort_index(inplace=True)

    bl_t   = bl.rename(columns={model_col: "vBL"}).assign(t_bl=bl.index).reset_index(drop=True)
    mlwv_t = mlwv.rename(columns={model_col: "vMLWV"}).assign(t_mlwv=mlwv.index).reset_index(drop=True)
    mlhf_t = mlhf.rename(columns={model_col: "vMLHF"}).assign(t_mlhf=mlhf.index).reset_index(drop=True)

    tol = pd.Timedelta(tolerance)

    # Merge-asof to the same reference clock
    tmp = pd.merge_asof(ref_t, bl_t,   left_on="t_ref", right_on="t_bl",   direction="nearest", tolerance=tol)
    tmp = pd.merge_asof(tmp,  mlwv_t,  left_on="t_ref", right_on="t_mlwv", direction="nearest", tolerance=tol)
    out = pd.merge_asof(tmp,  mlhf_t,  left_on="t_ref", right_on="t_mlhf", direction="nearest", tolerance=tol)

    out = (
        out.drop(columns=["t_bl", "t_mlwv", "t_mlhf"])
           .set_index("t_ref")
           .rename(columns={ref_col: ref_label})
           [[ref_label, "vBL", "vMLWV", "vMLHF"]]
           .dropna(subset=["vBL", "vMLWV", "vMLHF"])
    )
    return out

def _metrics(obs, mod):
    """Return dict: N, RMSE, R2, r, KGE."""
    obs = np.asarray(obs, dtype=float)
    mod = np.asarray(mod, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(mod)
    if ok.sum() < 2:
        return {"N": int(ok.sum()), "RMSE": np.nan, "R2": np.nan, "r": np.nan, "KGE": np.nan}
    o, m = obs[ok], mod[ok]
    N = o.size
    RMSE = rmse(o, m)
    R2 = r2_score(o, m)
    r, _ = pearsonr(o, m)
    alpha = np.nanstd(m, ddof=1) / np.nanstd(o, ddof=1) if np.nanstd(o, ddof=1) != 0 else np.nan
    beta  = np.nanmean(m) / np.nanmean(o) if np.nanmean(o) != 0 else np.nan
    KGE = 1.0 - np.sqrt((r-1.0)**2 + (alpha-1.0)**2 + (beta-1.0)**2) if np.isfinite([r, alpha, beta]).all() else np.nan
    return {"N": N, "RMSE": RMSE, "R2": R2, "r": r, "KGE": KGE}

def compute_mean_std_30min_JJA(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Filter to JJA, group by 30-min bins (hour, minute) across years,
    compute mean & std, and return a plotting-friendly frame.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Keep only June–July–August
    df = df[df.index.month.isin([6, 7, 8])]

    # Drop NaNs in the target column (mean/std ignore NaN anyway, this just removes empty bins)
    df = df.dropna(subset=[col])

    # Explicit hour & minute columns to avoid reset_index name collisions
    df = df.assign(hour=df.index.hour, minute=df.index.minute)

    # Group by hour:minute only (30-min resolution assumed)
    df_stats = (
        df.groupby(['hour', 'minute'])[col]
          .agg(['mean', 'std'])
          .reset_index()
          .sort_values(['hour', 'minute'])
    )

    # Synthetic datetime for plotting (same day for all; year=2000 to keep it simple)
    df_stats['time_of_day'] = pd.to_datetime(
        dict(year=2000, month=6, day=21, hour=df_stats['hour'], minute=df_stats['minute'])
    )
    return df_stats

# ---- Keep only JJA (June, July, August) ----
def subset_jja(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("subset_jja expects a DatetimeIndex.")
    return df.loc[df.index.month.isin([6, 7, 8])]



# ---------------- Paths & inputs ----------------
insitu_forcing = "/home/khanalp/data/processed/input_pystemmus/v3"
ICOS_location  = "/home/khanalp/data/ICOS2025"
station_details = "/home/khanalp/code/PhD/preprocessICOSdata/csvs/01_stationinfo_L0_ICOS2025_with_heights.csv"
modeloutput_dir = "/home/khanalp/paper01/final/"
figure_dir = "/home/khanalp/code/PhD/daytoday/figures/"
MODIS_path = "/home/khanalp/data/MODISLST/"
filtered_stations = '/home/khanalp/code/PhD/daytoday/figures/EBR/FilteredStations.csv'


df_station_details = pd.read_csv(station_details, index_col=0)
df_station_details_selected = df_station_details[df_station_details["IGBP_short_name"].isin(["ENF", "GRA"])] # only ENF and GRA.
df_ECOSTRESS = pd.read_csv("/home/khanalp/data/ECOSTRESSLSTPaper01-ECO-L2T-LSTE-002-results.csv") # ECOSTRESS temperature data.

surftemp_col_names = ['simulation_number', 'year', 'DOY', 'Ta', 'Ts(1)', 'Ts(2)', 'Tcave', 'Tsave'] # for surface temp columns.
model_versions = ['vbigleaf', 'vMLWV', 'vMLHF'] # Need to adopt to vMLHF later. 
emissivity_value = 0.98  # for in-situ LST
metrics_results = []     # collect ALL stations here


df_filtered_stations = pd.read_csv(filtered_stations)

# ---- ensure every station is written, even if skipped later
def add_nan_metrics(station, landuse, refs=("INSITU","MODIS","ECOSTRESS")):
    for ref in refs:
        for model in ("vBL","vML"):
            metrics_results.append({
                "station": station, "landuse": landuse, "reference": ref, "model": model,
                "N": np.nan, "RMSE": np.nan, "R2": np.nan, "r": np.nan, "KGE": np.nan
            })

# ---------------- Main loop ----------------
for station in df_filtered_stations['station'].unique():
    print(f"Processing station: {station}")

    # land-use label for this station
    landuse = df_station_details_selected.loc[
        df_station_details_selected['station_name'] == station, 'IGBP_short_name'
    ].values[0]

    # ---------- forcing ----------
    try:
        forcing_file = [f for f in os.listdir(insitu_forcing) if station in f][0]
        ds_forcing = xr.open_dataset(os.path.join(insitu_forcing, forcing_file))
    except IndexError:
        print(f"Forcing file not found for {station}")
        add_nan_metrics(station, landuse)
        continue

    # ---------- MODIS ----------
    try:
        MODIS_LST_file = [f for f in os.listdir(MODIS_path) if station in f][0]
        df_MODIS_LST = pd.read_csv(os.path.join(MODIS_path, MODIS_LST_file))
    except IndexError:
        print(f"MODIS file not found for {station}")
        add_nan_metrics(station, landuse)
        continue

    # ---------- in-situ ----------
    insitu_path = os.path.join(ICOS_location, station, f"ICOSETC_{station}_FLUXNET_HH_L2.csv")
    if not os.path.exists(insitu_path):
        print(f"In-situ file not found for {station}")
        add_nan_metrics(station, landuse)
        continue

    df_insitu = pd.read_csv(insitu_path)
    df_insitu['time'] = pd.to_datetime(df_insitu['TIMESTAMP_START'], format='%Y%m%d%H%M')
    df_insitu.set_index('time', inplace=True)
    df_insitu.replace(-9999, np.nan, inplace=True)

    # required insitu columns check
    required_cols = ['LW_OUT', 'LW_IN_F_MDS', 'LW_IN_F_MDS_QC']
    if not all(col in df_insitu.columns for col in required_cols):
        print(f"Missing required in-situ columns for {station}, skipping...")
        add_nan_metrics(station, landuse)
        continue

    # compute in-situ LST (°C) where QC==0
    valid_mask = (
        df_insitu['LW_OUT'].notna() &
        df_insitu['LW_IN_F_MDS'].notna() &
        (df_insitu['LW_IN_F_MDS_QC'] == 0)
    )
    df_insitu['LST'] = np.nan
    df_insitu.loc[valid_mask, 'LST'] = temperature_from_lw_out(
        df_insitu.loc[valid_mask, 'LW_OUT'].values,
        df_insitu.loc[valid_mask, 'LW_IN_F_MDS'].values,
        emissivity_value
    ) - 273.15

    # ---------- model outputs ----------
    station_dir = os.path.join(modeloutput_dir, landuse, station, "output")
    try:
        subdirs = [d for d in os.listdir(station_dir) if re.match(rf"{station}_\d{{4}}-\d{{2}}-\d{{2}}-\d", d)]
    except FileNotFoundError:
        print(f"Output directory not found for {station}")
        add_nan_metrics(station, landuse)
        continue

    if not subdirs:
        print(f"No simulation subdirectories found for {station}")
        add_nan_metrics(station, landuse)
        continue

    subdir_path = os.path.join(station_dir, subdirs[0])
    leaftemp_file_vBL = os.path.join(subdir_path,model_versions[0],"csvs","leaftemp.csv")
    leaftemp_file_vMLWV = os.path.join(subdir_path,model_versions[1],"csvs","leaftemp.csv")
    leaftemp_file_vMLHF = os.path.join(subdir_path,model_versions[2],"leaftemp.csv")
    surftemp_file_vBL = os.path.join(subdir_path,model_versions[0],"csvs","surftemp.csv")
    surftemp_file_vMLWV = os.path.join(subdir_path,model_versions[1],"csvs","surftemp.csv")
    surftemp_file_vMLHF = os.path.join(subdir_path,model_versions[2],"surftemp.csv")
    gap_file_vBL = os.path.join(subdir_path,model_versions[0],"csvs","gap.csv")
    gap_file_vMLWV = os.path.join(subdir_path,model_versions[1],"csvs","gap.csv")
    gap_file_vMLHF = os.path.join(subdir_path,model_versions[2],"gap.csv")
    
    required_files = [leaftemp_file_vBL, leaftemp_file_vMLWV, leaftemp_file_vMLHF, surftemp_file_vBL, surftemp_file_vMLWV, surftemp_file_vMLHF, gap_file_vBL, gap_file_vMLWV, gap_file_vMLHF]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Missing required model files for {station}:")
        for mf in missing_files:
            print(f"  - {mf}")
        add_nan_metrics(station, landuse)
        continue

    # read csvs
    df_leaftemp_vBL = pd.read_csv(leaftemp_file_vBL)
    df_surftemp_vBL = pd.read_csv(surftemp_file_vBL)
    df_gap_vBL      = pd.read_csv(gap_file_vBL)
    df_leaftemp_vMLWV = pd.read_csv(leaftemp_file_vMLWV)
    df_surftemp_vMLWV = pd.read_csv(surftemp_file_vMLWV)
    df_gap_vMLWV      = pd.read_csv(gap_file_vMLWV)
    df_leaftemp_vMLHF = pd.read_csv(leaftemp_file_vMLHF, skiprows = 2,  header = None)
    df_surftemp_vMLHF = pd.read_csv(surftemp_file_vMLHF)
    df_gap_vMLHF      = pd.read_csv(gap_file_vMLHF, skiprows = 2, header = None)

    df_surftemp_vBL.columns = surftemp_col_names
    df_surftemp_vMLWV.columns = surftemp_col_names
    
    df_surftemp_vMLHF = df_surftemp_vMLHF.iloc[1:]
    

    # add time index from forcing
    for df in [df_leaftemp_vBL, df_surftemp_vBL, df_gap_vBL,
               df_leaftemp_vMLWV, df_surftemp_vMLWV, df_gap_vMLWV,
               df_leaftemp_vMLHF, df_surftemp_vMLHF, df_gap_vMLHF]:
        df['time'] = pd.to_datetime(ds_forcing.time.values)
        df.set_index('time', inplace=True)

    # LST for both model versions (same math you used)
    Sunlit_BL = pd.concat([df_leaftemp_vBL.iloc[:, 0:30], df_surftemp_vBL["Ts(2)"]], axis=1).to_numpy(dtype=float)
    Shaded_BL = pd.concat([df_leaftemp_vBL.iloc[:, 30:60], df_surftemp_vBL["Ts(1)"]], axis=1).to_numpy(dtype=float)
    Ps_BL = df_gap_vBL.iloc[:, 0:31].to_numpy(dtype=float)
    W = np.full((31, 1), 1.0/31, dtype=float)
    Mix_BL = (Ps_BL * Sunlit_BL) + ((1.0 - Ps_BL) * Shaded_BL)
    LST_BL = (Mix_BL @ W).ravel()
    df_surftemp_vBL["LST"] = pd.Series(LST_BL, index=df_surftemp_vBL.index)

    Sunlit_MLWV = pd.concat([df_leaftemp_vMLWV.iloc[:, 0:30], df_surftemp_vMLWV["Ts(2)"]], axis=1).to_numpy(dtype=float)
    Shaded_MLWV = pd.concat([df_leaftemp_vMLWV.iloc[:, 30:60], df_surftemp_vMLWV["Ts(1)"]], axis=1).to_numpy(dtype=float)
    Ps_MLWV = df_gap_vMLWV.iloc[:, 0:31].to_numpy(dtype=float)
    Mix_MLWV = (Ps_MLWV * Sunlit_MLWV) + ((1.0 - Ps_MLWV) * Shaded_MLWV)
    LST_MLWV = (Mix_MLWV @ W).ravel()
    df_surftemp_vMLWV["LST"] = pd.Series(LST_MLWV, index=df_surftemp_vMLWV.index)
    
    Sunlit_MLHF = pd.concat([df_leaftemp_vMLHF.iloc[:, 0:30], df_surftemp_vMLHF["Ts(2)"]], axis=1).to_numpy(dtype=float)
    Shaded_MLHF = pd.concat([df_leaftemp_vMLHF.iloc[:, 30:60], df_surftemp_vMLHF["Ts(1)"]], axis=1).to_numpy(dtype=float)
    Ps_MLHF = df_gap_vMLHF.iloc[:, 0:31].to_numpy(dtype=float)
    Mix_MLHF = (Ps_MLHF * Sunlit_MLHF) + ((1.0 - Ps_MLHF) * Shaded_MLHF)
    LST_MLHF = (Mix_MLHF @ W).ravel()
    df_surftemp_vMLHF["LST"] = pd.Series(LST_MLHF, index=df_surftemp_vMLHF.index)

    # ---------- ECOSTRESS (°C) ----------
    df_ECOSTRESS_selected_station = df_ECOSTRESS[
        (df_ECOSTRESS['ID'] == station) & (df_ECOSTRESS['ECO_L2T_LSTE_002_cloud'] == 0)
    ].copy()
    
    df_ECOSTRESS_selected_station['Date'] = pd.to_datetime(df_ECOSTRESS_selected_station['Date'], utc=True)
    df_ECOSTRESS_selected_station['Date'] = df_ECOSTRESS_selected_station['Date'].dt.tz_convert('Europe/Berlin')
    df_ECOSTRESS_selected_station = df_ECOSTRESS_selected_station.set_index('Date')
    df_ECOSTRESS_selected_station.index = df_ECOSTRESS_selected_station.index.tz_localize(None)
    df_ECOSTRESS_LST_filtered = df_ECOSTRESS_selected_station.dropna(subset=['ECO_L2T_LSTE_002_LST']).copy()
    df_ECOSTRESS_LST_filtered.loc[:, 'ECO_L2T_LSTE_002_LST'] = df_ECOSTRESS_LST_filtered['ECO_L2T_LSTE_002_LST'] - 273.15
    df_ECOSTRESS_avg = (
        df_ECOSTRESS_LST_filtered
        .groupby(df_ECOSTRESS_LST_filtered.index)
        .agg({'ECO_L2T_LSTE_002_LST':'mean', 'ECOSTRESS_Tile': pd.Series.nunique})
        .rename(columns={'ECO_L2T_LSTE_002_LST':'LST_C', 'ECOSTRESS_Tile':'no_of_tiles'})
    )
    
    

    # ---------- MODIS (day+night) ----------
    base = df_MODIS_LST.assign(datetime=lambda d: pd.to_datetime(d['datetime']))
    day_df = (
        base.assign(
            td=lambda d: pd.to_timedelta(d['Day_view_time'], unit='h'),
            timestamp=lambda d: d['datetime'] + d['td'],
            LST_C=lambda d: d['LST_Day_C'],
            view='day'
        )[['timestamp','LST_C','view']]
    )
    night_df = (
        base.assign(
            td=lambda d: pd.to_timedelta(d['Night_view_time'], unit='h'),
            timestamp=lambda d: d['datetime'] + d['td'],
            LST_C=lambda d: d['LST_Night_C'],
            view='night'
        )[['timestamp','LST_C','view']]
    )
    df_modis_long = (pd.concat([day_df, night_df], ignore_index=True)
                     .dropna(subset=['timestamp','LST_C'])
                     .sort_values('timestamp', kind='mergesort')
                     .set_index('timestamp')
                     .sort_index())

    # Filtered copies
    df_surftemp_vBL_JJA = subset_jja(df_surftemp_vBL)
    df_surftemp_vMLWV_JJA = subset_jja(df_surftemp_vMLWV)
    df_surftemp_vMLHF_JJA = subset_jja(df_surftemp_vMLHF)
    df_insitu_JJA       = subset_jja(df_insitu)
    df_modis_long_JJA   = subset_jja(df_modis_long)
    df_ecostress_JJA    = subset_jja(df_ECOSTRESS_avg)

    # ---------- align model vs references (30 min) ----------
    df_modis_model_aligned = align_four(
    df_modis_long_JJA, "LST_C",
    df_surftemp_vBL_JJA, df_surftemp_vMLWV_JJA, df_surftemp_vMLHF_JJA,
    model_col="LST", tolerance="30min", ref_label="MODIS"
    )

    df_ecostress_model_aligned = align_four(
        df_ecostress_JJA, "LST_C",
        df_surftemp_vBL_JJA, df_surftemp_vMLWV_JJA, df_surftemp_vMLHF_JJA,
        model_col="LST", tolerance="30min", ref_label="ECOSTRESS"
    )

    df_insitu_model_aligned = align_four(
        df_insitu_JJA, "LST",
        df_surftemp_vBL_JJA, df_surftemp_vMLWV_JJA, df_surftemp_vMLHF_JJA,
        model_col="LST", tolerance="30min", ref_label="INSITU"
    )
     

    # ---------- figure dirs ----------
    # png_dir = os.path.join(figure_dir, landuse, station, "png")
    pdf_dir = os.path.join(figure_dir, landuse, station, "pdf")
    # os.makedirs(png_dir, exist_ok=True); 
    os.makedirs(pdf_dir, exist_ok=True)
    fname_tag = f"{station}_LST"

    # ---------- time series (LST + Ts(1)/Ts(2)) ----------
    fig_ts, ax_ts = plt.subplots(figsize=(12, 4))
    # model LST
    ax_ts.plot(df_surftemp_vBL.index, df_surftemp_vBL["LST"].values, label='vBL LST', color='tab:blue', linestyle='-', linewidth=0.6)
    ax_ts.plot(df_surftemp_vMLWV.index, df_surftemp_vMLWV["LST"].values, label='vMLWV LST', color='tab:orange', linestyle='-', linewidth=0.6)
    ax_ts.plot(df_surftemp_vMLHF.index, df_surftemp_vMLHF["LST"].values, label='vMLHF LST', color='tab:green', linestyle='-', linewidth=0.6)
    # soil temps
    # ax_ts.plot(df_surftemp_vBL.index, df_surftemp_vBL["Ts(1)"].values, label='vBL Ts(1)', linestyle='--', linewidth=0.6)
    # ax_ts.plot(df_surftemp_vBL.index, df_surftemp_vBL["Ts(2)"].values, label='vBL Ts(2)', linestyle='--', linewidth=0.6)
    # ax_ts.plot(df_surftemp_vML.index, df_surftemp_vML["Ts(1)"].values, label='vML Ts(1)', linestyle=':', linewidth=0.6)
    # ax_ts.plot(df_surftemp_vML.index, df_surftemp_vML["Ts(2)"].values, label='vML Ts(2)', linestyle=':', linewidth=0.6)
    # in-situ LST
    ax_ts.plot(df_insitu.index, df_insitu['LST'].values, label='In-situ LST', color='black', linestyle='-', linewidth=0.8, alpha=0.6)

    ax_ts.set_xlabel('Time')
    ax_ts.set_title(f" {station}, {landuse}")
    ax_ts.legend(frameon=False, loc="upper left")
    fig_ts.savefig(os.path.join(pdf_dir, f"timeseries_{fname_tag}.pdf"),
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_ts)

    # ---------- scatter: three panels ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
    datasets = [
        (df_insitu_model_aligned,    "INSITU",    "In-situ"),
        (df_modis_model_aligned,     "MODIS",     "MODIS"),
        (df_ecostress_model_aligned, "ECOSTRESS", "ECOSTRESS")
    ]

    for ax, (df_aligned, ref_label, title) in zip(axes, datasets):
        ref_col = df_aligned.columns[0]
        x = df_aligned[ref_col].values
        y_bl = df_aligned["vBL"].values
        y_mlWV = df_aligned["vMLWV"].values
        y_mlHF = df_aligned["vMLHF"].values

        # dynamic axis
        stack = np.concatenate([x, y_bl, y_mlWV, y_mlHF])
        stack = stack[np.isfinite(stack)]
        lo, hi = (np.nanpercentile(stack, [1, 99]) if stack.size else (0, 1))
        pad = 0.05 * max(hi - lo, 1.0); lo, hi = lo - pad, hi + pad

        # 1:1 line + points
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="gray", alpha=0.8)
        ax.scatter(x, y_bl, s=16, alpha=0.6, color="blue")
        ax.scatter(x, y_mlWV, s=16, alpha=0.6, color="orange")
        ax.scatter(x, y_mlHF, s=16, alpha=0.6, color="green")

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel(f"{ref_label} LST (°C)")
        if ax is axes[0]:
            ax.set_ylabel("Model LST (°C)")

        # metrics
        m_bl = _metrics(x, y_bl)
        m_mlWV = _metrics(x, y_mlWV)
        m_mlHF = _metrics(x, y_mlHF)
        txt_bl = (f"vBL:\nN={m_bl['N']}\nRMSE={m_bl['RMSE']:.1f}\nR²={m_bl['R2']:.3f}\n"
                  f"r={m_bl['r']:.3f}\nKGE={m_bl['KGE']:.3f}")
        txt_mlWV = (f"vMLWV:\nN={m_mlWV['N']}\nRMSE={m_mlWV['RMSE']:.1f}\nR²={m_mlWV['R2']:.3f}\n"
                  f"r={m_mlWV['r']:.3f}\nKGE={m_mlWV['KGE']:.3f}")
        txt_mlHF = (f"vMLHF:\nN={m_mlHF['N']}\nRMSE={m_mlHF['RMSE']:.1f}\nR²={m_mlHF['R2']:.3f}\n"
                  f"r={m_mlHF['r']:.3f}\nKGE={m_mlHF['KGE']:.3f}")

        ax.text(0.02, 0.98, txt_bl, transform=ax.transAxes, va="top", ha="left",
                fontsize=8, color="blue",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        ax.text(0.98, 0.02, txt_mlWV, transform=ax.transAxes, va="bottom", ha="right",
                fontsize=8, color="orange",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        ax.text(0.98, 0.98, txt_mlHF, transform=ax.transAxes, va="top", ha="right",
                fontsize=8, color="green",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # collect metrics rows
        metrics_results.append({
            "station": station, "landuse": landuse, "reference": ref_label, "model": "vBL",
            "N": m_bl["N"], "RMSE": m_bl["RMSE"], "R2": m_bl["R2"], "r": m_bl["r"], "KGE": m_bl["KGE"]
        })
        metrics_results.append({
            "station": station, "landuse": landuse, "reference": ref_label, "model": "vMLWV",
            "N": m_mlWV["N"], "RMSE": m_mlWV["RMSE"], "R2": m_mlWV["R2"], "r": m_mlWV["r"], "KGE": m_mlWV["KGE"]
        })
        metrics_results.append({
            "station": station, "landuse": landuse, "reference": ref_label, "model": "vMLHF",
            "N": m_mlHF["N"], "RMSE": m_mlHF["RMSE"], "R2": m_mlHF["R2"], "r": m_mlHF["r"], "KGE": m_mlHF["KGE"]
        })


    fig.suptitle(f"{station}, {landuse}", fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(pdf_dir, f"scatterplot_{station}_LST_JJA.pdf"),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # ===== Compute stats for each dataset =====
    df_stats_insitu_JJA = compute_mean_std_30min_JJA(df_insitu[['LST']], 'LST')
    df_stats_vBL_JJA    = compute_mean_std_30min_JJA(df_surftemp_vBL[['LST']], 'LST')
    df_stats_vMLWV_JJA    = compute_mean_std_30min_JJA(df_surftemp_vMLWV[['LST']], 'LST')
    df_stats_vMLHF_JJA    = compute_mean_std_30min_JJA(df_surftemp_vMLHF[['LST']], 'LST')
    
    fig, ax = plt.subplots(figsize=(6, 6))

    # In-situ
    ax.plot(df_stats_insitu_JJA['time_of_day'], df_stats_insitu_JJA['mean'],
            color='black', label='In-situ (mean)')
    ax.fill_between(df_stats_insitu_JJA['time_of_day'],
                    df_stats_insitu_JJA['mean'] - df_stats_insitu_JJA['std'],
                    df_stats_insitu_JJA['mean'] + df_stats_insitu_JJA['std'],
                    color='black', alpha=0.15, label='')

    # vBL
    ax.plot(df_stats_vBL_JJA['time_of_day'], df_stats_vBL_JJA['mean'],
            color='tab:blue', label='vBL (mean)')
    ax.fill_between(df_stats_vBL_JJA['time_of_day'],
                    df_stats_vBL_JJA['mean'] - df_stats_vBL_JJA['std'],
                    df_stats_vBL_JJA['mean'] + df_stats_vBL_JJA['std'],
                    color='tab:blue', alpha=0.15, label='')

    # vMLWV
    ax.plot(df_stats_vMLWV_JJA['time_of_day'], df_stats_vMLWV_JJA['mean'],
            color='tab:orange', label='vMLWV (mean)')
    ax.fill_between(df_stats_vMLWV_JJA['time_of_day'],
                    df_stats_vMLWV_JJA['mean'] - df_stats_vMLWV_JJA['std'],
                    df_stats_vMLWV_JJA['mean'] + df_stats_vMLWV_JJA['std'],
                    color='tab:orange', alpha=0.15, label='')
    
    # vMLHF
    ax.plot(df_stats_vMLHF_JJA['time_of_day'], df_stats_vMLHF_JJA['mean'],
            color='tab:green', label='vMLHF (mean)')
    ax.fill_between(df_stats_vMLHF_JJA['time_of_day'],
                    df_stats_vMLHF_JJA['mean'] - df_stats_vMLHF_JJA['std'],
                    df_stats_vMLHF_JJA['mean'] + df_stats_vMLHF_JJA['std'],
                    color='tab:green', alpha=0.15, label='')

    # Format x-axis to show only time
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # Rotate x-axis tick labels
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=8)

    # Smaller labels & title
    ax.set_xlabel("Time of Day (JJA)", fontsize=9)
    ax.set_ylabel("LST (°C)", fontsize=9)
    ax.set_title(f"{station}, {landuse} - JJA", fontsize=10)

    # Grid & legend
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    
   
    plt.tight_layout()
    fig.savefig(os.path.join(pdf_dir, f"{station}_LST_JJA.pdf"),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)


    fig.tight_layout()
    plt.show()


    #pdf
    fig_pdf, ax_pdf = plt.subplots(figsize=(6, 6))  
    sns.kdeplot(df_insitu['LST'].values, color="black", fill=True, linewidth=1, label="In-situ", ax=ax_pdf)
    sns.kdeplot(df_surftemp_vBL["LST"].values, color="tab:blue", fill=True, linewidth=1, label="vBL", ax=ax_pdf)
    sns.kdeplot(df_surftemp_vMLWV["LST"].values, color="tab:orange", fill=True, linewidth=1, label="vMLWV", ax=ax_pdf)
    sns.kdeplot(df_surftemp_vMLHF["LST"].values, color="tab:green", fill=True, linewidth=1, label="vMLHF", ax=ax_pdf)
    # Labels & title
    title = f"{station}, {landuse}" if landuse else station
    ax_pdf.set_title(f"{title} – LST", fontsize=10)
    ax_pdf.set_ylabel("Density", fontsize=9) 
    ax_pdf.set_xlabel(f"degreeC", fontsize  =9)
    ax_pdf.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.show()
    fig_pdf.savefig(os.path.join(pdf_dir, f"pdf_{station}_LST.pdf"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_pdf)
    
    # df_err = pd.DataFrame({
    #         "obs": df_insitu['LST'].values,
    #         "err_BL": df_surftemp_vBL["LST"].values - df_insitu['LST'].values,
    #         "err_ML": df_surftemp_vML["LST"].values - df_insitu['LST'].values
    #     })
    
    # fig_error, ax_error = plt.subplots(figsize=(6, 4))

    # sns.kdeplot(df_err["err_BL"], fill=True, color="tab:blue", label="vBL", ax=ax_error)
    # sns.kdeplot(df_err["err_ML"], fill=True, color="tab:orange", label="vML", ax=ax_error)
    # title = f"{station}, {landuse}" if landuse else station
    # ax_error.axvline(0, color="black", linestyle="--")
    # ax_error.set_xlabel("Error (Model – In-situ)")
    # ax_error.set_ylabel("Probability Density")
    # ax_error.set_title(f"{title} – LST, error distribution", fontsize=10)
    # ax_error.legend()
    # plt.tight_layout()
    # plt.show()
    # fig_error.savefig(os.path.join(png_dir, f"pdf_{station}_LST_error.png"),
    #             dpi=600, bbox_inches='tight', facecolor='white')
    # plt.close(fig_error)
     
     


# ---------------- Save metrics for ALL stations ----------------
df_metrics_scatter = pd.DataFrame(metrics_results)
out_csv = "/home/khanalp/code/PhD/daytoday/csvs/comparison_LST_metrics_JJA_for_filtered_station.csv"
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df_metrics_scatter.to_csv(out_csv, index=False)
print(f"Saved all metrics to {out_csv}")
