"""
This script connects to an FTP server of TROPOMISIF, navigates through its directory structure, 
and downloads all files while preserving the directory hierarchy locally.

Functions:
    download_files(ftp, path, local_path):
        Recursively downloads files and directories from the specified FTP server path 
        to the local directory, maintaining the directory structure.

Variables:
    ftp_host (str): The hostname of the FTP server.
    ftp_dir (str): The root directory on the FTP server to start downloading from.
    local_dir (str): The local root directory where files will be saved.

Usage:
    - Ensure the FTP server details (ftp_host and ftp_dir) are correct.
    - Specify the local directory (local_dir) where files should be saved.
    - Run the script to download files from the FTP server.
"""

import os
import ftplib

# FTP server details
ftp_host = 'ftp.sron.nl'
ftp_dir = '/open-access-data-2/TROPOMI/tropomi/sif/v2.1/l2b/'

# Local directory to save files (root folder)
local_dir = '/home/khanalp/data/TROPOMISIF/'

# Create local directory if it doesn't exist
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

def download_files(ftp, path, local_path):
    """Recursively download files from FTP server while preserving directory structure."""
    # Change to the specified path on the FTP server
    ftp.cwd(path)

    # List the contents of the current directory
    items = ftp.nlst()

    for item in items:
        try:
            # Try to change into directory
            ftp.cwd(item)
            # If successful, it's a directory, so create the directory locally
            new_local_path = os.path.join(local_path, item)
            if not os.path.exists(new_local_path):
                os.makedirs(new_local_path)
            # Recursively download files from this directory
            download_files(ftp, ftp.pwd(), new_local_path)
            # Go back to the parent directory on the FTP server
            ftp.cwd('..')
        except ftplib.error_perm:
            # If permission error occurs, it's a file, so download it
            local_file = os.path.join(local_path, item)
            with open(local_file, 'wb') as f:
                ftp.retrbinary('RETR ' + item, f.write)
            print(f"Downloaded {item}")

# Connect to the FTP server
ftp = ftplib.FTP(ftp_host)
ftp.login()

# Start downloading files from the root directory, preserving the structure
download_files(ftp, ftp_dir, local_dir)

# Close the connection
ftp.quit()

print("Download complete!")
