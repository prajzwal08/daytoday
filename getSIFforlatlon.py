"""
Script to extract Solar-Induced Fluorescence (SIF) data for specific latitude and longitude coordinates 
(station locations) from TROPOMI NetCDF files and save the results to CSV files.

Modules:
    - os: For file and directory operations.
    - pandas: For data manipulation and analysis.
    - numpy: For numerical computations.
    - datetime: For handling dates and times.
    - netCDF4: For working with NetCDF files.
    - logging: For logging messages.

Constants:
    - path_TROPOMI_SIF: Path to the directory containing TROPOMI SIF data.
    - path_station_info: Path to the CSV file containing station information.
    - path_output: Path to the directory where processed data will be saved.
    - log_file: Path to the log file with a timestamp.

Workflow:
    1. Configure logging to track script execution.
    2. Read station information from a CSV file.
    3. Iterate over each station to process SIF data:
        - Skip processing if the output file for the station already exists.
        - Extract latitude and longitude for the station.
        - Traverse year and month folders in the TROPOMI SIF directory.
        - For each NetCDF file:
            - Extract relevant data (latitude, longitude, SIF_743, time, delta_time).
            - Find the closest data point to the station's coordinates.
            - Calculate the SIF value and corresponding timestamp.
            - Append results to a list.
    4. Save the processed data for each station to a CSV file.
    5. Log errors and progress throughout the script execution.

Exceptions:
    - Handles and logs errors during file reading, NetCDF processing, and data extraction.

Output:
    - CSV files containing SIF data for each station, sorted by time.
    - Log file documenting the script's execution and any errors encountered.
"""

# Import necessary libraries
import os  # For file and directory operations
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
from datetime import datetime, timedelta  # For handling dates and times
from netCDF4 import Dataset  # For working with NetCDF files
import logging  # For logging messages

# Define the path to the TROPOMI SIF data directory
path_TROPOMI_SIF = "/home/khanalp/data/TROPOMISIF"

# Define the path to the station information CSV file
path_station_info = "/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv"

# Define the output directory for processed data
path_output = "/home/khanalp/data/processed/tropomisif/"

# Get the current time for logging purposes
current_time = datetime.now()

# Define the log file path with a timestamp
log_file = f"/home/khanalp/logs/getSIFforlatlon_{current_time.strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging settings
logging.basicConfig(
    filename=log_file,  # Log file path
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log message format
)

# Log the start of the script
logging.info("Script started.")

# Define the base time for converting time variables in NetCDF files
base_time = datetime(2010, 1, 1, 0, 0, 0)

try:
    # Log the start of reading the station info CSV file
    logging.info("Reading station info CSV file.")

    # Read the station information from the CSV file into a DataFrame
    df_station_info = pd.read_csv(path_station_info)

    # Log the number of stations loaded from the CSV file
    logging.info(f"Loaded station info for {len(df_station_info)} stations.")

    # Iterate over each station in the station info DataFrame
    for station_name in df_station_info.station_name:
        # Define the output file path for the current station
        output_file = os.path.join(path_output, f"{station_name}_SIF.csv")

        # Check if the output file already exists
        if os.path.exists(output_file):
            # Log that the file exists and skip processing for this station
            logging.info(f"Output file {output_file} already exists. Skipping station: {station_name}")
            continue

        # Log the start of processing for the current station
        logging.info(f"Processing station: {station_name}")

        # Get the information for the current station
        station_info = df_station_info[df_station_info['station_name'] == station_name]

        # Extract the latitude and longitude of the station
        latitude = station_info['latitude'].values[0]
        longitude = station_info['longitude'].values[0]

        # Log the latitude and longitude of the station
        logging.info(f"Station {station_name} has latitude {latitude} and longitude {longitude}.")

        # Initialize a list to store results for the current station
        results = []

        # Get the list of year folders in the TROPOMI SIF data directory
        years_folder = [f for f in os.listdir(path_TROPOMI_SIF) if os.path.isdir(os.path.join(path_TROPOMI_SIF, f))]

        # Log the number of year folders found
        logging.info(f"Found {len(years_folder)} year folders in TROPOMI SIF path.")

        # Iterate over each year folder
        for year in years_folder:
            # Define the path to the current year folder
            year_path = os.path.join(path_TROPOMI_SIF, year)

            # Check if the year path is a directory
            if os.path.isdir(year_path):
                # Log the processing of the current year folder
                logging.info(f"Processing year folder: {year}")

                # Iterate over each month folder in the year folder
                for month in os.listdir(year_path):
                    # Define the path to the current month folder
                    month_path = os.path.join(year_path, month)

                    # Check if the month path is a directory
                    if os.path.isdir(month_path):
                        # Log the processing of the current month folder
                        logging.info(f"Processing month folder: {month}")

                        # Iterate over each file in the month folder
                        for file in os.listdir(month_path):
                            # Check if the file is a NetCDF file
                            if file.endswith(".nc"):
                                # Define the full path to the NetCDF file
                                file_path = os.path.join(month_path, file)

                                # Log the processing of the current NetCDF file
                                logging.info(f"Processing file: {file_path}")

                                try:
                                    # Open the NetCDF file
                                    ds_TROPOMI = Dataset(file_path, 'r')

                                    # Access the 'PRODUCT' group in the NetCDF file
                                    product_group = ds_TROPOMI.groups['PRODUCT']

                                    # Extract latitude, longitude, SIF_743, delta_time, and time variables
                                    latitudes = product_group.variables['latitude'][:]
                                    longitudes = product_group.variables['longitude'][:]
                                    sif_743 = product_group.variables['SIF_743'][:]
                                    delta_time = product_group.variables['delta_time'][:]
                                    time = product_group.variables['time'][:]

                                    # Calculate distances between the station and all data points
                                    distances = np.sqrt((latitudes - latitude)**2 + (longitudes - longitude)**2)

                                    # Find the index of the closest data point
                                    closest_index = np.argmin(distances)

                                    # Get the SIF value for the closest data point
                                    sif_value = sif_743[closest_index]

                                    # Convert the base time and time variable to a datetime object
                                    converted_time = base_time + timedelta(seconds=float(time[0]))

                                    # Add the delta time to the converted time
                                    delta_time_value = int(delta_time[closest_index])
                                    converted_time += timedelta(milliseconds=delta_time_value)

                                    # Append the results for the current file
                                    results.append({
                                        "file": file_path,  # File path
                                        "time": converted_time,  # Converted time
                                        "SIF_743": sif_value  # SIF value
                                    })
                                except Exception as e:
                                    # Log any errors that occur while processing the file
                                    logging.error(f"Error processing file {file_path}: {e}")

        # Create a DataFrame from the results
        df_results = pd.DataFrame(results)

        # Sort the results by time and reset the index
        df_results = df_results.sort_values(by="time").reset_index(drop=True)

        # Save the results to a CSV file
        df_results.to_csv(output_file, index=False)

        # Log that the results have been saved
        logging.info(f"Results saved to {output_file}")

except Exception as e:
    # Log any errors that occur during the script execution
    logging.error(f"An error occurred: {e}")

# Log the end of the script
logging.info("Script finished.")
