import os
import numpy as np
import csv
import glob

# User input for n
n = int(input("Enter the value of Station: "))

# Specify the folder containing the CSV files
folder_path = "/home/gautam/Downloads/cappy-definitive-edition/step4/sce1a_output/rssi"  # Change this to your folder path

# Use glob to get all CSV files and sort them numerically based on filenames
csv_files = sorted(glob.glob(os.path.join(folder_path, "rssi_*.csv")), key=lambda x: int(x.split('_')[-1].split('.')[0]))

# List to store the results for output CSV
rssi_results = []

# Iterate through all CSV files in the folder
for file_path in csv_files:
    filename = os.path.basename(file_path)
    #print(f"Processing file: {filename}")

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [list(map(float, row)) for row in reader]  # Convert to float

    data = np.array(data)  # Convert to NumPy array

    # Extract station values based on n
    station_values = []
    for i in range(n):
        station_values.append(data[:, (i + 1)::(n + 1)])  # Collect all station columns

    station_values = np.hstack(station_values)  # Concatenate all station columns

    # Compute average of station values
    station_averages = np.mean(station_values, axis=1)
    final_rssi = 0.2 * ((station_averages + 90) / 60)

    # Append the result to the list
    rssi_results.append([final_rssi])

    #print(f"Final RSSI for {filename}: {final_rssi}")

# Write the results to a new CSV file
output_file_path = "/home/gautam/Downloads/cappy-definitive-edition/step5/final_rssi.csv"  # Change to desired output path

with open(output_file_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["rssi"])  # Write header
    writer.writerows(rssi_results)  # Write the collected data

print(f"Final RSSI values saved to {output_file_path}")
 # Concatenate all station columns

    # Compute average of station values

