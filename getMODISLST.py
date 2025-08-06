import os
import ee
import logging 
import pandas as pd
from datetime import datetime

# Initialize Earth Engine
ee.Initialize()

# paths
station_info_path = "/home/khanalp/code/PhD/preprocessICOSdata/csvs/01_stationinfo_L0_ICOS2025_with_heights.csv"
output_dir = "/home/khanalp/paper01/MODISLST/"
os.makedirs(output_dir, exist_ok=True)

# Set up logging
log_file = os.path.join(output_dir, "MODIS_LST_download.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s", 
    filemode='w'  # overwrite each run; change to 'a' to append
)


#Read station details from CSV
df_station_details = pd.read_csv(station_info_path, index_col=0)
df_station_details['start_date'] = pd.to_datetime(df_station_details['start_date']).dt.strftime('%Y-%m-%d')
df_station_details['end_date'] = pd.to_datetime(df_station_details['end_date']).dt.strftime('%Y-%m-%d')

# Process: scale LST to Celsius, view times to hours
def process(img):
    lst_day_c = img.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('LST_Day_C')
    lst_night_c = img.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename('LST_Night_C')
    return img.addBands(lst_day_c).addBands(lst_night_c)


# for each station, get MODIS LST data
for station in df_station_details['station_name'].unique():
    try:
        logging.info(f"Processing station: {station}")
        # Define point and time range
        longitude = df_station_details[df_station_details['station_name'] == station]['longitude'].values[0]
        latitude = df_station_details[df_station_details['station_name'] == station]['latitude'].values[0]
        start_date = df_station_details[df_station_details['station_name'] == station]['start_date'].values[0]  
        end_date = df_station_details[df_station_details['station_name'] == station]['end_date'].values[0]
        
        logging.debug(f"{station} - Location: ({latitude}, {longitude}), Period: {start_date} to {end_date}")
        point = ee.Geometry.Point([longitude, latitude])

        # Select all relevant bands
        collection = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterDate(start_date, end_date) \
            .filterBounds(point) \
            .select([
                'LST_Day_1km', 'Day_view_time',
                'LST_Night_1km', 'Night_view_time'
            ])
            
        logging.debug(f"{station} - Image collection filtered, starting to apply processing.")        
        processed = collection.map(process)
        
        # Sample at the point (MODIS scale = 1000 m)
        scale = 1000
        results = processed.getRegion(point, scale).getInfo()
        
        if not results or len(results) < 2:
            logging.warning(f"{station} - No MODIS data found for the specified time range and location. Skipping...")
            continue
        
        # Convert to pandas DataFrame
        header = results[0]
        records = results[1:]
        df = pd.DataFrame(records, columns=header)
        # Convert 'time' from milliseconds since epoch to datetime and set as index
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        df.drop(columns=['time'], inplace=True)
        df = df.set_index('datetime')
        # Convert 'Day_view_time' from tenths of hours to hours
        df['Day_view_time'] = df['Day_view_time'].astype(float) * 0.1
        # Convert 'Day_view_time' from tenths of hours to hours
        df['Night_view_time'] = df['Night_view_time'].astype(float) * 0.1
        # Keep only relevant columns
        df = df[['LST_Day_C', 'Day_view_time','LST_Night_C', 'Night_view_time']]
        # Save the cleaned DataFrame
        output_path = f"{output_dir}{station}_MODIS_LST.csv"
        df.to_csv(output_path)
        logging.info(f"Saved LST data to {output_path}")
    except Exception as e:
        logging.error(f"Error processing station {station}: {e}")
     
    