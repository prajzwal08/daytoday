# ========================================================================
# Purpose of the Code:
# ========================================================================
# This script processes Sentinel-2 L2A satellite data for each station listed in
# the CSV file containing station metadata (including latitude, longitude, 
# and other details). For each station:
# 1. A bounding box polygon is created around the station with a 100-meter buffer.
# 2. The time range is split into 6-month periods.
# 3. The STAC catalog API is queried for Sentinel-2 L2A data within the bounding 
#    box and the specific time intervals.
# 4. Metadata such as datetime and platform serial identifier is collected.
# 5. The metadata for each station is saved as a CSV file in the specified 
#    output directory.

# ========================================================================
# Detailed Workflow:
# ========================================================================
# 1. **Station Data Filtering**:
#    - Read station details from a CSV file.
#    - Extract latitude and longitude for each station.
# 2. **Bounding Box Creation**:
#    - Generate a 100m buffer square around each station to define a GeoJSON polygon.
# 3. **Time Range Handling**:
#    - Split the provided time range into 6-month intervals.
# 4. **Sentinel-2 L2A Data Search**:
#    - Use the STAC API to search for data within the bounding box and the time intervals.
# 5. **Metadata Collection**:
#    - Collect metadata (datetime, platform serial identifier) for each Sentinel-2 item.
# 6. **Output**:
#    - Save the metadata as a CSV file for each station.
# ========================================================================

# Logging - Log the main processes for tracking
import logging


# ========================================================================
# Import necessary libraries
# ========================================================================
import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pystac_client import Client
from datetime import datetime, date

# ========================================================================
# File paths and catalog connection
# ========================================================================
path_station_details = "/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv"
path_sentinel2 = "/home/khanalp/data/sentinel/l2a/"
output_location = "/home/khanalp/data/sentinel/l2a/metadata/"

# Set up logging to output both to console and a file
log_file = "/home/khanalp/logs/sentinel_metadata_processing.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # Save log messages to a file
        logging.StreamHandler()         # Also output log messages to the console
    ]
)

# Now, log messages will be saved to the specified log file
logging.info("This is an info message")

# Connect to the CDSE STAC catalog
catalog = Client.open("https://catalogue.dataspace.copernicus.eu/stac")

# Load the station metadata from the CSV file
df_station_details = pd.read_csv(path_station_details)

# ========================================================================
# Define the function to create a bounding box polygon around a station
# ========================================================================
def get_bbox_polygon(latitude: float, longitude: float, buffer_m: float) -> dict:
    """
    Return a GeoJSON polygon representing a square buffer around a point.

    Parameters:
        latitude (float): Latitude of the center point.
        longitude (float): Longitude of the center point.
        buffer_m (float): Buffer distance in meters.

    Returns:
        dict: GeoJSON polygon representing the bounding box.
    """
    km2deg = 1.0 / 111  # Approx. conversion from km to degrees
    buffer_deg = (buffer_m / 1000.0) * km2deg

    west = longitude - buffer_deg
    east = longitude + buffer_deg
    south = latitude - buffer_deg
    north = latitude + buffer_deg

    return {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [west, north],
            [east, north],
            [east, south],
            [west, south]  # Close the polygon
        ]]
    }

# ========================================================================
# Define the function to split time range into 6-month intervals
# ========================================================================
def split_time_range_by_six_months(time_range: list[str]) -> list[tuple[str, str]]:
    """
    Split the provided time range into 6-month intervals.
    """
    start = datetime.strptime(time_range[0], "%Y-%m-%d").date()
    end = datetime.strptime(time_range[1], "%Y-%m-%d").date()

    result = []
    current_start = start

    # First handle the case where we are in the first half of the first year
    if current_start.month <= 6:
        current_end = date(current_start.year, 6, 30)  # End of June
        result.append((current_start.isoformat(), current_end.isoformat()))
        current_start = date(current_start.year, 7, 1)  # Start of second half
    else:
        current_end = date(current_start.year, 12, 31)  # End of December
        result.append((current_start.isoformat(), current_end.isoformat()))
        current_start = date(current_start.year + 1, 1, 1)  # Move to next year

    # Loop for the remaining full years
    while current_start.year < end.year:
        # First half of the year (Jan-Jun)
        current_end = date(current_start.year, 6, 30)
        result.append((current_start.isoformat(), current_end.isoformat()))
        current_start = date(current_start.year, 7, 1)  # Start of second half
        
        # Second half of the year (Jul-Dec)
        current_end = date(current_start.year, 12, 31)
        result.append((current_start.isoformat(), current_end.isoformat()))
        current_start = date(current_start.year + 1, 1, 1)  # Move to next year

    # Handle the last partial period (if any)
    if current_start <= end:
        result.append((current_start.isoformat(), end.isoformat()))

    return result

# ========================================================================
# Main processing loop
# ========================================================================
# Split the time range into 6-month intervals
time_range = ["2015-07-04", "2024-03-31"]
six_month_ranges = split_time_range_by_six_months(time_range)

# Loop through each station and process the data
for station in df_station_details.station_name:
    all_items = []  
    logging.info(f"Processing station: {station}")  # Log the current station being processed

    # Filter the DataFrame to get metadata for the current station
    station_data = df_station_details[df_station_details.station_name == station]

    # Extract latitude and longitude for the station
    latitude = station_data.latitude.values[0]
    longitude = station_data.longitude.values[0]

    # Create a bounding box polygon (GeoJSON) with a 100m buffer around the station
    bbox = get_bbox_polygon(latitude, longitude, 100)
    
    # Find Sentinel-2 files for the current station
    file = [f for f in os.listdir(path_sentinel2) if station in f and f.endswith(".nc")]
    ds_sentinel2 = xr.open_dataset(os.path.join(path_sentinel2, file[0]))
    time = ds_sentinel2.t.values
    coords = bbox['coordinates'][0]

    # Get the minimum and maximum longitude and latitude for the bounding box
    lons = [pt[0] for pt in coords]
    lats = [pt[1] for pt in coords]
    box = [min(lons), min(lats), max(lons), max(lats)]
    
    # Loop through the 6-month intervals and search for data in each
    for date_start, date_end in six_month_ranges:
        logging.info(f"Searching for data between {date_start} and {date_end}")
        
        # Search for Sentinel-2 L2A data
        search = catalog.search(
            collections=["SENTINEL-2"],
            bbox=box,
            datetime=f"{date_start}/{date_end}",  # Use the current date range
        )

        items = search.item_collection()
        items = sorted(
            {item.id: item for item in items}.values(),
            key=lambda x: x.properties["datetime"]
        )

        # Append metadata for each found item
        for item in items:
            all_items.append({
                "station": station,
                "datetime": item.properties["datetime"],
                "platform": item.properties.get("platformSerialIdentifier"),
            })

    # Save the collected metadata to a CSV file
    df_all_items = pd.DataFrame(all_items)
    output_filename = f"{station}_{time_range[0]}_{time_range[1]}.csv"
    output_path = os.path.join(output_location, output_filename)
    df_all_items.to_csv(output_path, index=False)
    
    logging.info(f"Metadata saved for {station} to {output_path}")
    

   