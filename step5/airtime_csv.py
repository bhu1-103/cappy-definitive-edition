import os
import numpy as np
import csv
import glob

# Specify the folder containing the CSV files
folder_path = "/home/gautam/Downloads/cappy-definitive-edition/step4/sce1a_output/airtime"  # Change this to your folder path

# Use glob to get all CSV files and sort them numerically based on filenames
csv_files = sorted(glob.glob(os.path.join(folder_path, "airtime_*.csv")), key=lambda x: int(x.split('_')[-1].split('.')[0]))

# List to store the results for output CSV
airtime_results = []

# Iterate through all CSV files in the folder
for file_path in csv_files:
    filename = os.path.basename(file_path)

    data = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')  # Use correct delimiter
        for row in reader:
            for cell in row:
                values = cell.split(',')  # Split values by comma
                for val in values:
                    val = val.strip()  # Remove extra spaces
                    if val.lower() == 'inf':
                        data.append(np.nan)  # Replace 'inf' with NaN
                    elif val:
                        data.append(float(val))

    data = np.array(data)  # Convert to NumPy array

    # Compute average without considering NaN values
    overall_average = np.nanmean(data)  # nanmean ignores NaNs in mean calculation
    final_airtime = 0.2 * overall_average / 100

    # Append the result to the list
    airtime_results.append([final_airtime])

# Write the results to a new CSV file
output_file_path = "/home/gautam/Downloads/cappy-definitive-edition/step5/final_airtime.csv"  # Change to desired output path

with open(output_file_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["airtime"])  # Write header
    writer.writerows(airtime_results)  # Write the collected data

print(f"Final airtime values saved to {output_file_path}")
