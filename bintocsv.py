import os 
import re
import pandas as pd
import numpy as np
import logging

# Define paths 
modeloutput_dir = "/home/khanalp/paper01/final/"
binfileinfo_path = "/home/khanalp/code/PhD/daytoday/csvs/binfilesinfoSTEMMUSSCOPE.csv"
log_summary_path = "/home/khanalp/code/PhD/daytoday/csvs/log_summary.csv"
station_info_path = "/home/khanalp/code/PhD/preprocessICOSdata/csvs/01_stationinfo_L0_ICOS2025_with_heights.csv"
log_path = "/home/khanalp/paper01/logs/bin_to_csv_conversion.log"

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode='w'  # overwrite each run; change to 'a' to append
)
log = logging.getLogger()

# Read CSV files 
df_binfilesinfo = pd.read_csv(binfileinfo_path)
df_log_summary = pd.read_csv(log_summary_path)
df_station_info = pd.read_csv(station_info_path, index_col=0)

# Filter stations based on land use
selected_landuse = ["ENF", "GRA"]
df_station_selected = df_station_info[df_station_info["IGBP_short_name"].isin(selected_landuse)]

# Find stations where both 'vMLWV' and 'vbigleaf' have the same 'timestep'
matching_stations = (
    df_log_summary
    .pivot_table(index=['station_name', 'timestep'], columns='version', aggfunc='size', fill_value=0)
    .reset_index()
)

# Keep only rows where both versions exist for the same station and timestep
matching_stations = matching_stations[(matching_stations.get('vMLWV', 0) > 0) & (matching_stations.get('vbigleaf', 0) > 0)]

# Get the set of (station_name, timestep) pairs to keep
valid_pairs = set(zip(matching_stations['station_name'], matching_stations['timestep']))

# Filter df_log_summary to keep only those with both versions and same timestep
df_log_summary_filtered = df_log_summary[
    df_log_summary.apply(lambda row: (row['station_name'], row['timestep']) in valid_pairs, axis=1)
]

# Find removed stations
removed_stations = set(df_station_selected['station_name']) - set(df_log_summary_filtered['station_name'])

# In modeloutput_dir, the output directory for each station is stored  i{n following way:\
    # modeloutput_dir/{landuse}/{station}/output/{station}_2025-07-25-{id}/{version} 
    # landuse can be ENF or GRA
    #station comes from station
    # id is just a 1 digit number
    # version can be vMLWV or vbigleaf.
    # I want to get output dir for each station and version.
    # Find .bin files in each version directory and convert them to CSV.
for station in df_log_summary_filtered['station_name'].unique():
    try: 
        landuse = df_station_selected[df_station_selected['station_name'] == station]['IGBP_short_name'].values[0]
        nrow = df_log_summary_filtered[df_log_summary_filtered['station_name'] == station]['modelruntime'].values[0]
        station_dir = os.path.join(modeloutput_dir, landuse, station, "output")
        
        if not os.path.isdir(station_dir):
            log.warning(f"Directory missing for station: {station} at path: {station_dir}")
            continue
        log.info(f"Processing station: {station}, landuse: {landuse}, output dir: {station_dir}")
        
        # Find all subdirectories matching {station}_2025-07-25-{id}
        for subdir in os.listdir(station_dir):
            match = re.match(rf"{station}_\d{{4}}-\d{{2}}-\d{{2}}-(\d)", subdir)
            if match:
                subdir_path = os.path.join(station_dir, subdir)
                for version in ["vMLWV", "vbigleaf"]:
                    version_dir = os.path.join(subdir_path, version)
                    if not os.path.isdir(version_dir):
                        log.warning(f"Missing version dir: {version_dir}")
                        continue
                    log.info(f"Processing version: {version}, path: {version_dir}")
                    # List all .bin files here H
                    bin_files = [f for f in os.listdir(version_dir) if f.endswith('.bin')] 
                    
                    if not bin_files:
                        log.warning(f"No .bin files found in {version_dir}")
                        continue
                    
                    for bin_file in bin_files:  
                        bin_file_path = os.path.join(version_dir, bin_file)
                        log.info(f"Found .bin file: {bin_file_path}")
                        # for corresponding bin file, i need to get the ncol from df_binfilesinfo
                        try: 
                            ncol = int(df_binfilesinfo[df_binfilesinfo['Filename'] == bin_file]['ncol'])
                        except:
                            log.error(f"Could not find ncol for {bin_file} in binfilesinfo")
                            continue
                        
                        #  Read binary data
                        data = np.fromfile(bin_file_path, dtype=np.float64)
                        # Step 2: Check size and reshape
                        expected_size = nrow * ncol
                        if data.size != int(expected_size):
                            log.error(f"Expected {expected_size} values ({nrow}x{ncol}), but got {data.size}")
                            continue 
                        data = data.reshape((nrow, ncol))
                        df = pd.DataFrame(data)
                        # Make sure output directory exists and save there
                        csv_dir = os.path.join(version_dir, "csvs")
                        os.makedirs(csv_dir, exist_ok=True)

                        output_csv = os.path.join(csv_dir, f"{bin_file[:-4]}.csv")
                        df.to_csv(output_csv, index=False)
                        log.info(f"Converted {bin_file} to CSV: {output_csv}")
                        # Break after processing the first bin file for each version
    except Exception as e:    
        log.exception(f"Unexpected error while processing station {station}")
        continue           
    


