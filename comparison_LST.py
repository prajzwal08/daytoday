# Import necessary libraries
import pandas as pd
import xarray as xr
import os
import sys
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
from scipy.stats import pearsonr

# local utils (unused here but kept as in your codebase)
sys.path.append('/home/khanalp/code/PhD/utils')

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
    lw_out = np.asarray(lw_out)
    lw_in = np.asarray(lw_in)
    if np.any(lw_out < 0):
        raise ValueError("LW_OUT values must be non-negative.")
    return ((lw_out - ((1-emissivity) * lw_in)) / (emissivity * sigma)) ** 0.25

def compute_metrics(df, obs_col, model_col='model_LST'):
    """
    Compute R2 and RMSE between observed and modeled LST.
    
    Args:
        df (pd.DataFrame): DataFrame containing observed and model columns
        obs_col (str): Name of observed LST column (Planet, MODIS, ECOSTRESS)
        model_col (str): Name of model LST column
    
    Returns:
        dict: {'R2': value, 'RMSE': value}
    """
    o = df[obs_col].values
    m = df[model_col].values
    r2 = r2_score(o, m)
    RMSE  = np.sqrt(mean_squared_error(o, m))
    N = o.size
    r, _ = pearsonr(o, m)
    return {'N': N, 'R2': r2, 'RMSE': RMSE, 'r': r}


    # ---------------- Paths & inputs ----------------
insitu_forcing = "/home/khanalp/data/processed/input_pystemmus/v3"
ICOS_location  = "/home/khanalp/data/ICOS2025"
station_details = "/home/khanalp/code/PhD/preprocessICOSdata/csvs/01_stationinfo_L0_ICOS2025_with_heights.csv"
modeloutput_dir = "/home/khanalp/code/PhD/daytoday/csvs/LSTs"
figure_dir = "/home/khanalp/code/PhD/daytoday/figures/LSTcomparison"
MODIS_path = "/home/khanalp/data/MODISLST/"
filtered_stations = '/home/khanalp/code/PhD/daytoday/figures/EBR/FilteredStations.csv'
planet_LST_path = '/home/khanalp/data/Planet_data_LST/'

df_station_details = pd.read_csv(station_details, index_col=0)
df_station_details_selected = df_station_details[df_station_details["IGBP_short_name"].isin(["ENF", "GRA"])] # only ENF and GRA.
df_ECOSTRESS = pd.read_csv("/home/khanalp/data/ECOSTRESSLSTPaper01-ECO-L2T-LSTE-002-results.csv") # ECOSTRESS temperature data.


emissivity_value = 0.98  # for in-situ LST
records = []     # collect ALL stations here


df_filtered_stations = pd.read_csv(filtered_stations)

# ---- ensure every station is written, even if skipped later
def add_nan_metrics(station, landuse, refs=("INSITU","MODIS","ECOSTRESS")):
    for ref in refs:
        for model in ("vBL","vML"):
            records.append({
                "station": station, "landuse": landuse, "reference": ref, "model": model,
                "N": np.nan, "RMSE": np.nan, "R2": np.nan, "r": np.nan, "KGE": np.nan
            })


all_metrics = []

for station in df_filtered_stations['station'].unique():
    print(f"Processing station: {station}")
     
    try: 
        # land-use label for this station
        landuse = df_station_details_selected.loc[
            df_station_details_selected['station_name'] == station, 'IGBP_short_name'
        ].values[0]
        
        latitude = df_station_details_selected.loc[
            df_station_details_selected['station_name'] == station, 'latitude'
        ].values[0]
        
        longitude = df_station_details_selected.loc[
            df_station_details_selected['station_name'] == station, 'longitude'
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
        
        # read in-situ data and add index. 
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
        
        # compute in-situ LST (°C) where QC==0 (only when measured and both LW_IN and LW_OUT are present)
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

        # Model
        modeloutput_file = [f for f in os.listdir(modeloutput_dir) if station in f][0]
        df_model_output = pd.read_csv(os.path.join(modeloutput_dir, modeloutput_file))
        df_model_output.index = pd.to_datetime(df_model_output['time'])
        df_model_output.drop(columns=['time'], inplace=True)
        # Determine model time range
        model_start = df_model_output.index.min()
        model_end = df_model_output.index.max()

        
        #---------- ECOSTRESS (°C) ----------
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
        
        # Merge based on nearest timestamp
        # This is a pandas function designed to merge time-series data where timestamps don’t exactly match.
        # It finds the closest key (timestamp) on the right DataFrame for each key on the left DataFrame.
        # Unlike a regular merge (which matches exact values), merge_asof aligns nearest values in time order.
        df_ECOSTRESS_combined = pd.merge_asof(
            df_ECOSTRESS_avg[['LST_C']],      # take only ECOSTRESS LST
            df_model_output[['LST']],       # take only model LST
            left_index=True,
            right_index=True,
            direction='nearest'
        )

        # Rename columns
        df_ECOSTRESS_combined.columns = ['ECOSTRESS_LST', 'model_LST']
        
        # --- Merge with nearest in-situ SW_IN_F_MDS and LST ---
        df_filtered = pd.merge_asof(
            df_ECOSTRESS_combined,
            df_insitu[['SW_IN_F_MDS', 'LST']],  # include both radiation and in-situ LST
            left_index=True,
            right_index=True,
            direction='nearest'
        )

        # --- Filter for daytime (SW_IN_F_MDS >= 10 W/m²) ---
        df_ECOSTRESS_model_day = df_filtered[df_filtered['SW_IN_F_MDS'] >= 10]

        # --- Drop unneeded columns and NaNs ---
        df_ECOSTRESS_model_day = (
            df_ECOSTRESS_model_day
            .drop(columns=['SW_IN_F_MDS'])
            .dropna(subset=['ECOSTRESS_LST', 'model_LST', 'LST'])  # ensure all 3 available
            )
            

        df_ECOSTRESS_model_day.columns = ['ECOSTRESS_LST', 'model_LST', 'insitu_LST']
        
        
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
        df_modis_day = df_modis_long[df_modis_long['view']=='day'].copy()
        
        # Unlike a regular merge (which matches exact values), merge_asof aligns nearest values in time order.
        df_modis_model = pd.merge_asof(
            df_modis_day[['LST_C']],      # take only ECOSTRESS LST
            df_model_output[['LST']],       # take only model LST
            left_index=True,
            right_index=True,
            direction='nearest'
        )
        df_modis_model.columns = ['MODIS_LST', 'model_LST']
        # --- Step 2: Merge with nearest in-situ LST only ---
        df_modis_model = pd.merge_asof(
            df_modis_model,
            df_insitu[['LST']],               # in-situ LST only
            left_index=True,
            right_index=True,
            direction='nearest'
        )
        df_modis_model.columns = ['MODIS_LST', 'model_LST', 'insitu_LST']
        # --- Step 3: Drop missing values ---
        df_modis_model = df_modis_model.dropna(subset=['MODIS_LST', 'model_LST', 'insitu_LST'])
        
        # Planet LST
        file_planet_LST = [f for f in os.listdir(planet_LST_path) if 'ASC' in f and str(longitude) in f and str(latitude) in f][0]
        df_planet_LST_day = pd.read_csv(os.path.join(planet_LST_path, file_planet_LST), skiprows = 6)
        df_planet_LST_day.index = pd.to_datetime(df_planet_LST_day['date'])
        df_planet_LST_day.drop(columns=['date'], inplace=True)
        # Convert index to datetime at 13:30
        df_planet_LST_day.index = pd.to_datetime(df_planet_LST_day.index) + pd.Timedelta(hours=13, minutes=30)
        df_planet_LST_day.columns = ['LST']
        df_planet_LST_day['LST'] = df_planet_LST_day['LST'] - 273.15  # Convert from K to °C
        
        # Merge with nearest timestamp in model_output
        df_planet_model = pd.merge_asof(
            df_planet_LST_day,
            df_model_output[['LST']],  # model LST
            left_index=True,
            right_index=True,
            direction='nearest'
        )
        
        df_planet_model.columns = ['Planet_LST', 'model_LST']
        # --- Step 2: Merge with nearest in-situ LST only ---
        df_planet_model = pd.merge_asof(
            df_planet_model,
            df_insitu[['LST']],               # in-situ LST only
            left_index=True,
            right_index=True,
            direction='nearest'
        )
        df_planet_model.columns = ['Planet_LST', 'model_LST', 'insitu_LST']
        # --- Step 3: Drop missing values ---
        df_planet_model = df_planet_model.dropna(subset=['Planet_LST', 'model_LST', 'insitu_LST'])
        
        # Constrain between model start and end date only. 
        df_modis_model = df_modis_model[(df_modis_model.index >= model_start) & (df_modis_model.index <= model_end)]
        df_planet_model = df_planet_model[(df_planet_model.index >= model_start) & (df_planet_model.index <= model_end)]
        df_ECOSTRESS_model_day = df_ECOSTRESS_model_day[(df_ECOSTRESS_model_day.index >= model_start) & (df_ECOSTRESS_model_day.index <= model_end)]
        
        # --- Model vs Satellite ---
        planet_metrics_model = compute_metrics(df_planet_model, 'Planet_LST', 'model_LST')
        modis_metrics_model = compute_metrics(df_modis_model, 'MODIS_LST', 'model_LST')
        ecostress_metrics_model = compute_metrics(df_ECOSTRESS_model_day, 'ECOSTRESS_LST', 'model_LST')

        # --- In-situ vs Satellite ---
        planet_metrics_insitu = compute_metrics(df_planet_model, 'Planet_LST', 'insitu_LST')
        modis_metrics_insitu = compute_metrics(df_modis_model, 'MODIS_LST', 'insitu_LST')
        ecostress_metrics_insitu = compute_metrics(df_ECOSTRESS_model_day, 'ECOSTRESS_LST', 'insitu_LST')


        
            # ---------------- Store metrics with station and landuse info ----------------
        metrics_station = {
            'station': station,
            'landuse': landuse,
            
            # --- Planet ---
            'Planet_N_model': planet_metrics_model['N'],
            'Planet_r_model': planet_metrics_model['r'],
            'Planet_R2_model': planet_metrics_model['R2'],
            'Planet_RMSE_model': planet_metrics_model['RMSE'],
            'Planet_N_insitu': planet_metrics_insitu['N'],
            'Planet_r_insitu': planet_metrics_insitu['r'],
            'Planet_R2_insitu': planet_metrics_insitu['R2'],
            'Planet_RMSE_insitu': planet_metrics_insitu['RMSE'],
            
            # --- MODIS ---
            'MODIS_N_model': modis_metrics_model['N'],
            'MODIS_r_model': modis_metrics_model['r'],
            'MODIS_R2_model': modis_metrics_model['R2'],
            'MODIS_RMSE_model': modis_metrics_model['RMSE'],
            'MODIS_N_insitu': modis_metrics_insitu['N'],
            'MODIS_r_insitu': modis_metrics_insitu['r'],
            'MODIS_R2_insitu': modis_metrics_insitu['R2'],
            'MODIS_RMSE_insitu': modis_metrics_insitu['RMSE'],
            
            # --- ECOSTRESS ---
            'ECOSTRESS_N_model': ecostress_metrics_model['N'],
            'ECOSTRESS_r_model': ecostress_metrics_model['r'],
            'ECOSTRESS_R2_model': ecostress_metrics_model['R2'],
            'ECOSTRESS_RMSE_model': ecostress_metrics_model['RMSE'],
            'ECOSTRESS_N_insitu': ecostress_metrics_insitu['N'],
            'ECOSTRESS_r_insitu': ecostress_metrics_insitu['r'],
            'ECOSTRESS_R2_insitu': ecostress_metrics_insitu['R2'],
            'ECOSTRESS_RMSE_insitu': ecostress_metrics_insitu['RMSE'],
        }
        all_metrics.append(metrics_station)
        
    
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

        # ---------- Planet ----------
        axes[0].plot(df_planet_model.index, df_planet_model['Planet_LST'], label='Planet LST', color = 'blue', linestyle='-')
        axes[0].plot(df_planet_model.index, df_planet_model['model_LST'], label='Model LST', color='black', linestyle='-')
        axes[0].plot(df_planet_model.index, df_planet_model['insitu_LST'], label='In-situ LST', color='red', linestyle='-', linewidth=2)
        axes[0].set_ylabel('LST (°C)')
        axes[0].set_title(f'Planet vs Model LST for {station}')
        axes[0].legend()
        axes[0].grid(True)

        # ---------- MODIS ----------
        axes[1].plot(df_modis_model.index, df_modis_model['MODIS_LST'], label='MODIS LST', color = 'blue', linestyle='-')
        axes[1].plot(df_modis_model.index, df_modis_model['model_LST'], label='Model LST', color='black', linestyle='-')
        axes[1].plot(df_modis_model.index, df_modis_model['insitu_LST'], label='In-situ LST', color='red', linestyle='-', linewidth=2)
        axes[1].set_ylabel('LST (°C)')
        axes[1].set_title(f'MODIS vs Model LST for {station}')
        axes[1].legend()
        axes[1].grid(True)

        # ---------- ECOSTRESS ----------
        axes[2].plot(df_ECOSTRESS_model_day.index, df_ECOSTRESS_model_day['ECOSTRESS_LST'], label='ECOSTRESS LST', color = 'blue',  linestyle='-')
        axes[2].plot(df_ECOSTRESS_model_day.index, df_ECOSTRESS_model_day['model_LST'], label='Model LST', color='black', linestyle='-')
        axes[1].plot(df_ECOSTRESS_model_day.index, df_ECOSTRESS_model_day['insitu_LST'], label='In-situ LST', color='red', linestyle='-', linewidth=2)
        axes[2].set_ylabel('LST (°C)')
        axes[2].set_title(f'ECOSTRESS vs Model LST for {station}')
        axes[2].legend()
        axes[2].grid(True)

        # ---------- Formatting ----------
        axes[2].set_xlabel('Time')
        plt.tight_layout()
        plt.show()
        # Save time series plot
        ts_plot_path = os.path.join(figure_dir, f'{station}_LST_timeseries.png')
        plt.savefig(ts_plot_path)
        plt.close()
        
        df_list = [df_planet_model, df_modis_model, df_ECOSTRESS_model_day]
        labels = ['Planet_LST', 'MODIS_LST', 'ECOSTRESS_LST']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex='col', sharey='row')

        for i, label in enumerate(labels):
            df_plot = df_list[i]

            # ---------- Row 1: Model vs Satellite ----------
            ax1 = axes[0, i]
            ax1.scatter(df_plot[label], df_plot['model_LST'], alpha=0.6)
            min_val = min(df_plot[label].min(), df_plot['model_LST'].min())
            max_val = max(df_plot[label].max(), df_plot['model_LST'].max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--')
            ax1.grid(True)
            
            metric_model = compute_metrics(df_plot, label, 'model_LST')
            ax1.text(
                0.05, 0.95,
                f"N = {metric_model['N']}\n"
                f"r = {metric_model['r']:.2f}\n"
                f"R² = {metric_model['R2']:.2f}\n"
                f"RMSE = {metric_model['RMSE']:.2f} °C",
                transform=ax1.transAxes,
                fontsize=10,
                va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7)
            )

            # ---------- Row 2: In-situ vs Satellite ----------
            ax2 = axes[1, i]
            if 'insitu_LST' in df_plot.columns:
                ax2.scatter(df_plot[label], df_plot['insitu_LST'], alpha=0.6, color='tab:orange')
                min_val2 = min(df_plot[label].min(), df_plot['insitu_LST'].min())
                max_val2 = max(df_plot[label].max(), df_plot['insitu_LST'].max())
                ax2.plot([min_val2, max_val2], [min_val2, max_val2], 'k--')
                ax2.grid(True)

                metric_insitu = compute_metrics(df_plot, label, 'insitu_LST')
                ax2.text(
                    0.05, 0.95,
                    f"N = {metric_insitu['N']}\n"
                    f"r = {metric_insitu['r']:.2f}\n"
                    f"R² = {metric_insitu['R2']:.2f}\n"
                    f"RMSE = {metric_insitu['RMSE']:.2f} °C",
                    transform=ax2.transAxes,
                    fontsize=10,
                    va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7)
                )
            else:
                ax2.text(0.5, 0.5, 'No In-situ Data', ha='center', va='center', transform=ax2.transAxes)

            if i == 0:
                ax1.set_ylabel('Model LST (°C)')
                ax2.set_ylabel('In-situ LST (°C)')
            ax1.set_xlabel(f'{label} (°C)')
            ax2.set_xlabel(f'{label} (°C)')

        # ---------- Set overall figure title ----------
        fig.suptitle(f'LST Comparison: Satellite, Model, and In-situ for {station} ({landuse})', fontsize=16, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
        plt.show()

        # Save scatter plot
        scatter_plot_path = os.path.join(figure_dir, f'{station}_LST_scatter.png')
        plt.savefig(scatter_plot_path)
        plt.close()


    except (IndexError, FileNotFoundError, ValueError) as e:
            print(f"Skipping station {station}: {e}")
            add_nan_metrics(station, landuse)  # optional: store NaN metrics
            continue
     
# Save all metrics to CSV
metrics_df_all = pd.DataFrame(all_metrics)
metrics_csv_path = os.path.join(figure_dir, 'all_stations_LST_metrics.csv')
metrics_df_all.to_csv(metrics_csv_path, index=False)
