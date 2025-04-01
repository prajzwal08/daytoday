"""
This script compares model output fluorescence data with TROPOMI SIF (Solar-Induced Fluorescence) observations.
It processes data for multiple stations, aligns timestamps, performs unit conversions, and generates time series
and scatter plots to visualize and evaluate the comparison.
"""
import os  # For file and directory operations
import pandas as pd  # For handling tabular data
import xarray as xr  # For working with NetCDF files
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
from datetime import datetime, timedelta  # For date and time operations
from netCDF4 import Dataset  # For working with NetCDF files
from sklearn.metrics import r2_score  # For calculating R-squared metric
import logging  # For logging script execution details

log_file = '/home/khanalp/code/PhD/daytoday/SIFcomparison/comparison_with_tropomi_sif.log'
# Configure logging
logging.basicConfig(
    filename= log_file,  # Log file name
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log message format
)

# Log the start of the script
logging.info("Script started: Comparison of Model Output and TROPOMI SIF")

# Define paths to input and output data
path_TROPOMI_SIF = "/home/khanalp/data/processed/tropomisif"  # Path to TROPOMI SIF data
path_model_output_nc = "/home/khanalp/data/processed/output_pystemmus"  # Path to model output NetCDF files
path_station_info = "/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv"  # Path to station info CSV
pystemmus_output_model_folder = "/home/khanalp/STEMMUS_SCOPE_model/STEMMUS_SCOPE_old/STEMMUS_SCOPE/ICOS_sites/"  # Path to model output folders
path_output_plot = "/home/khanalp/code/PhD/daytoday/SIFcomparison"  # Path to save plots

# Load station information
df_station_info = pd.read_csv(path_station_info)
# Initialize an empty list to store results
results = [] 
# Loop through each station to process and compare data
for station_name in df_station_info.station_name:
    try:
        # Extract station-specific information
        station_info = df_station_info[df_station_info['station_name'] == station_name]
        latitude = station_info['latitude'].values[0]
        longitude = station_info['longitude'].values[0]
        land_use = station_info['IGBP_short_name'].values[0]
        
        # Find the corresponding model output file for the station
        file_name = [f for f in os.listdir(path_model_output_nc) if station_name in f]
        path_model_output_station = [os.path.join(path_model_output_nc, f) for f in os.listdir(path_model_output_nc) if station_name in f]
        
        # create output folder for the station
        output_folder = os.path.join(path_output_plot, station_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        if not path_model_output_station:
            logging.warning(f"No model output file found for station {station_name}")
            continue
        
        # Open the model output NetCDF file
        ds_model_output = xr.open_dataset(path_model_output_station[0])
        
        # Extract the folder identifier from the file name
        folder_identifier = file_name[0].split('_')[1]
        
        # Find the corresponding station folder in the model output directory
        required_folder_station = [os.path.join(pystemmus_output_model_folder, f) for f in os.listdir(pystemmus_output_model_folder) if station_name in f]
        if not required_folder_station:
            logging.warning(f"No folder found for station {station_name} in pystemmus output folder")
            continue
        
        # Find the specific model run folder using the folder identifier
        required_model_run_folder = [os.path.join(required_folder_station[0], "output", f) for f in os.listdir(os.path.join(required_folder_station[0], "output")) if folder_identifier in f]
        if not required_model_run_folder:
            logging.warning(f"No model run folder found for station {station_name} with identifier {folder_identifier}")
            continue
        
        # Load fluorescence data from the model run folder
        df_fluoroscence_csv = pd.read_csv(
            os.path.join(required_model_run_folder[0], "fluorescence.csv"),
            skiprows=2,  # Skip the first two rows
            header=None  # No header in the file
        )
        # Update the headers to represent wavelengths from 640 to 850 nm
        df_fluoroscence_csv.columns = [f"{wavelength} nm" for wavelength in range(640, 851)]
        df_fluoroscence_csv.index = pd.to_datetime(ds_model_output.time.values)  # Align timestamps with model output
        
    
        # Find the corresponding TROPOMI SIF file for the station
        filename_tropomi_sif = [f for f in os.listdir(path_TROPOMI_SIF) if station_name in f]
        if not filename_tropomi_sif:
            logging.warning(f"No TROPOMI SIF file found for station {station_name}")
            continue
        
        # Load TROPOMI SIF data
        df_tropomi_sif = pd.read_csv(os.path.join(path_TROPOMI_SIF, filename_tropomi_sif[0]))
        df_tropomi_sif['time'] = pd.to_datetime(df_tropomi_sif['time']) 
        df_tropomi_sif.index = df_tropomi_sif['time'] # Convert time column to datetime
        df_tropomi_sif_filtered = df_tropomi_sif[ df_tropomi_sif.index <= df_fluoroscence_csv.index.max()]  # Filter data within the model output time range
        
        # Align timestamps between model output and TROPOMI SIF data
        df_fluoroscence_csv.index = pd.to_datetime(df_fluoroscence_csv.index)
        closest_times = df_fluoroscence_csv.index.get_indexer(df_tropomi_sif_filtered['time'], method='nearest')
        df_fluoroscence_csv_selected = df_fluoroscence_csv.iloc[closest_times]
        
        # Select fluorescence data for wavelengths 743-758 nm and calculate the mean
        df_fluoroscence_csv_selected_743_758 = df_fluoroscence_csv_selected.loc[:, "743 nm":"758 nm"]
        df_fluoroscence_csv_selected_743_758['mean'] = df_fluoroscence_csv_selected_743_758.mean(axis=1)
        
        # Generate time series plot
        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.plot(df_tropomi_sif_filtered.index, df_fluoroscence_csv_selected_743_758['mean'].values.flatten(), label='Model Output (743-758 nm)', color='blue', linestyle='-', linewidth=2)
        plt.plot(df_tropomi_sif_filtered.index, df_tropomi_sif_filtered['SIF_743'].values.flatten(), label='TROPOMI SIF (743 nm)', color='orange', linestyle='--', linewidth=2)

        # Add labels, title, and legend
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Fluorescence (mW/m²/sr/nm)', fontsize=14)
        plt.title('Comparison of Model Output and TROPOMI SIF', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Improve tick formatting
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)

        # Tight layout for better spacing
        plt.tight_layout()
        plt.title(f'Comparison of Model Output and TROPOMI SIF at station {station_name}({land_use})', fontsize=16)
        # Save the plot as a high-resolution image (optional)
        output_file = os.path.join(output_folder, f'Timeseries_{station_name}.png')
        plt.savefig(output_file, dpi=300)

        plt.show() 
        
        # Generate scatter plot
        plt.clf()
        # Remove NaN values from both datasets
        valid_indices = ~np.isnan(df_tropomi_sif_filtered['SIF_743'].values.flatten()) & ~np.isnan(df_fluoroscence_csv_selected_743_758['mean'].values.flatten())
        filtered_tropomi_sif = df_tropomi_sif_filtered['SIF_743'][valid_indices]
        filtered_model_fluorescence = df_fluoroscence_csv_selected_743_758['mean'][valid_indices]
        r2 = r2_score(filtered_tropomi_sif.values.flatten(), filtered_model_fluorescence.values.flatten())
        
        plt.figure(figsize=(8, 6))

        # Scatter plot
        plt.scatter(df_tropomi_sif_filtered['SIF_743'], df_fluoroscence_csv_selected_743_758['mean'], 
                    color='black', alpha=0.7, edgecolor='k', label='Data Points')

        
        # Add R-squared value and number of observations to the plot
        num_observations = np.sum(valid_indices)
        plt.text(0.05, 0.95, f'$R^2$: {r2:.2f}\nN: {num_observations}', 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
             bbox=dict(facecolor='white', alpha=0.5))

        # Add a 1:1 line for reference
        max_val = max(df_tropomi_sif_filtered['SIF_743'].max(), df_fluoroscence_csv_selected_743_758['mean'].max())
        plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='1:1 Line')

        # Add labels, title, and legend
        plt.xlabel('TROPOMI SIF  [mW/m²/sr/nm]', fontsize=14)
        plt.ylabel('Model Fluorescence  [mW/m²/sr/nm]', fontsize=14)
        plt.title(f'Scatter Plot of TROPOMI SIF vs Model Fluorescence\nStation: {station_name} ({land_use})', fontsize=16)
        plt.legend(fontsize=12)

        # Improve tick formatting
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.6)

        # Tight layout for better spacing
        plt.tight_layout()

        output_file = os.path.join(output_folder, f'Scatterplot_{station_name}.png')
        # Save the plot as a high-resolution image (optional)
        plt.savefig(output_file, dpi=300)

        plt.show()
        results.append({'station_name': station_name,
                        'land_use': land_use,
                        'r_squared': r2,
                        'num_observations': num_observations})
        logging.info(f"Completed station {station_name}: R^2 = {r2:.2f}, Observations = {num_observations}")
    except Exception as e:
        logging.error(f"An error occurred for station {station_name}: {e}")
    
# Convert results to a DataFrame and save it as a CSV file
results_df = pd.DataFrame(results)
results_output_file = os.path.join(path_output_plot, 'comparison_results.csv')
results_df.to_csv(results_output_file, index=False)

# Log the completion of the script
logging.info("Script completed successfully. Results saved to comparison_results.csv") 