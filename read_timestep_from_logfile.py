from pathlib import Path
import re
import csv

log_dir = Path("/home/khanalp/paper01/logs/")
output_csv = log_dir / "log_summary.csv"


# Updated regex with capture groups for station name and version
filename_pattern = re.compile(r"^([A-Za-z0-9\-]+)_2025-07-25-\d{4}_config_([A-Za-z0-9\-]+)\.log$")
timestep_pattern = re.compile(r"Warning: Timestep = (\d+)")

results = []

# Loop over matching files
for log_file in log_dir.glob("*.log"):
    match_filename = filename_pattern.match(log_file.name)
    if not match_filename:
        continue

    station_name = match_filename.group(1)
    version = match_filename.group(2)
    last_timestep = None

    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = timestep_pattern.search(line)
            if match:
                last_timestep = int(match.group(1))

    if last_timestep is not None:
        modelruntime = last_timestep - 1
        results.append({
            "filename": log_file.name,
            "station_name": station_name,
            "version": version,
            "timestep": last_timestep,
            "modelruntime": modelruntime
        })
    

# Write to CSV
with output_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "station_name", "version", "timestep", "modelruntime"])
    writer.writeheader()
    writer.writerows(results)

print(f"Saved summary to: {output_csv.resolve()}")

