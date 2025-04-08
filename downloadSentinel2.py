# -----------------------------------------------------------------------------------
# Script to retrieve Sentinel-2 L2A surface reflectance data from Copernicus Data Space Ecosystem
# for a list of stations, compute cloud-free mosaics using Scene Classification Layer (SCL),
# and calculate mean values for selected bands within a buffered area around each station.
# The results are intended for further analysis and comparison with ICOS station observations.
# -----------------------------------------------------------------------------------

import os 
import logging  # For logging steps and errors
import openeo  # OpenEO Python client for interacting with back-end processing services
import pandas as pd  # For handling tabular station information

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/home/khanalp/logs/sentinel_processing.log"),
        logging.StreamHandler()
    ]
)

# Path to CSV file containing station details (name, latitude, longitude, etc.)
path_station_details = "/home/khanalp/code/PhD/preprocessICOSdata/csvs/02_station_with_elevation_heightcanopy.csv"

# Output path for storing Sentinel-2 data (if needed for future use)
output_path = "/home/khanalp/data/sentinel/l2a"

# Connect to the OpenEO back-end hosted at Copernicus Data Space Ecosystem
logging.info("Connecting to OpenEO backend...")
connection = openeo.connect("openeofed.dataspace.copernicus.eu").authenticate_oidc()
logging.info("Connection established and authenticated.")

# Read the station metadata from the CSV into a pandas DataFrame
df_station_details = pd.read_csv(path_station_details)
logging.info(f"Loaded station metadata from: {path_station_details}")

def get_bbox_polygon(latitude: float, longitude: float, buffer_m: float) -> dict:
    """
    Return a GeoJSON polygon representing a square buffer around a point.

    Parameters:
        latitude (float): Latitude of center point.
        longitude (float): Longitude of center point.
        buffer_m (float): Buffer distance in meters.

    Returns:
        dict: GeoJSON polygon.
    """
    km2deg = 1.0 / 111  # Approx. conversion from kilometers to degrees latitude/longitude
    buffer_deg = (buffer_m / 1000.0) * km2deg  # Convert buffer from meters to degrees

    west = longitude - buffer_deg  # Left boundary of the box
    east = longitude + buffer_deg  # Right boundary of the box
    south = latitude - buffer_deg  # Bottom boundary
    north = latitude + buffer_deg  # Top boundary

    # Return the bounding box as a closed GeoJSON polygon
    return {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [west, north],
            [east, north],
            [east, south],
            [west, south]  # Close the polygon by repeating the first coordinate
        ]]
    }

def getBAP(scl, data):
    """
    Apply a cloud and vegetation mask to Sentinel-2 data using the Scene Classification Layer (SCL),
    and select Best Available Pixel (BAP) by masking out unwanted pixels.

    Parameters:
        scl: Scene Classification Layer data cube.
        data: Original Sentinel-2 surface reflectance cube.

    Returns:
        Data cube with masked (filtered) values.
    """
    # Create mask to exclude pixels where SCL is not equal to 4 (vegetation class only)
    mask = (scl != 4)

    # Apply the mask to the original data
    data_masked = data.mask(mask)

    # Return the masked data for further processing (e.g., spatial aggregation)
    return data_masked

# Define the time range for which Sentinel-2 data should be retrieved
time_range = ["2015-07-04", "2024-03-31"]

# Loop over each station name in the DataFrame
for station in df_station_details.station_name:
    try:
        logging.info(f"Processing station: {station}")
        print(station)  # Print the current station name

        # Filter the DataFrame to get metadata for the current station
        station_data = df_station_details[df_station_details.station_name == station]

        # Extract the latitude and longitude values
        latitude = station_data.latitude.values[0]
        longitude = station_data.longitude.values[0]

        # Create a bounding box polygon (GeoJSON) with 100m buffer around the station
        bbox_polygon = get_bbox_polygon(latitude, longitude, 100)

        # Load Sentinel-2 surface reflectance data (excluding SCL) from the back-end
        s2pre = connection.load_collection(
            "SENTINEL2_L2A",  # Name of the Sentinel-2 Level 2A product
            temporal_extent=time_range,  # Time range of interest
            spatial_extent=bbox_polygon,  # Bounding box around station
            bands=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09","B11", "B12"],  # Reflectance bands
            max_cloud_cover=90,  # Include images with up to 90% cloud cover
        )

        # Load the Scene Classification Layer (SCL) for cloud and land cover masking
        s2pre_scl = connection.load_collection(
            "SENTINEL2_L2A",  # Same collection
            temporal_extent=time_range,  # Same time range
            spatial_extent=bbox_polygon,  # Same bounding box
            bands=["SCL"],  # Only the SCL band
            max_cloud_cover=90,
        )

        logging.info(f"Data loaded for station: {station}")

        # Apply masking using SCL and get Best Available Pixel (BAP) image
        masked_image = getBAP(s2pre_scl, s2pre)
        logging.info(f"Applied cloud and vegetation mask for station: {station}")

        # Aggregate the masked image spatially over the bounding box using mean reducer
        mean_value = masked_image.aggregate_spatial(
            geometries=bbox_polygon,
            reducer=openeo.processes.mean
        )

        logging.info(f"Mean values computed for station: {station}")
        # Save the mean value to a to a nc file 
        # save result cube as JSON
        output_file = os.path.join(output_path, f"{station}_sentinel_l2a_{time_range[0]}_{time_range[1]}.nc")
        
        job = mean_value.create_job(title=f"Sentinel2_{station}_timeseries", out_format="netcdf")
        job.start()
        result = job.get_results()
        # Stop after the first station for testing/debugging
        # result.download_file("/home/khanalp/data/sentinel/l2a/" + f"{station}_sentinel_l2a_{time_range[0]}_{time_range[1]}.nc")
        # logging.info(f"Results downloaded for station: {station} at {output_file}")

    except Exception as e:
        logging.error(f"Error processing station {station}: {e}")

logging.info("Script completed.")

# After the jobs are completed, you can download the results
list_jobs = connection.list_jobs()
for job in list_jobs:
    job_id = job['id']
    job_title = job['title']
    job = connection.job(job_id)
    results = job.get_results()
    output_file = os.path.join(output_path, f"{job_title}_{time_range[0]}_{time_range[1]}.nc")
    results.download_file(output_file)
    


