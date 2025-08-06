# Import necessary modules
import pandas as pd
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('/home/khanalp/code/PhD/utils')
from utils import read_csv_file_with_station_name,temperature_from_lw_out

# Paths to different file
insitu_forcing = "/home/khanalp/data/processed/input_pystemmus/v2"
ICOS_location = "/home/khanalp/data/ICOS2020"
model_output_ml = "/home/khanalp/paper01/output/vMLWV/IT-Lav/v1"
model_output_bigleaf = "/home/khanalp/paper01/output/vbigleaf/IT-Lav/v1"
MODIS_LST = "/home/khanalp/paper01/output/LSTMODIS/IT-Lav_2004.csv"
