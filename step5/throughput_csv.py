import os
import numpy as np
import csv
import glob

# User input for n
n = int(input("Enter the value of station: "))

# Specify the folder containing the CSV files
folder_path = "/home/gautam/Downloads/cappy-definitive-edition/step4/sce1a_output/throughput"  # Change this to your folder path

# Use glob to get all CSV files and sort them numerically based on filenames
csv_files = sorted(glob.glob(os.path.join(folder_path, "throughput_*.csv")), key=lambda x: int(x.split('_')[-1].split('.')[0]))

# List to store the results for output CSV
throughput_results = []

# Iterate through all CSV files in the folder
for file_path in csv_files:
    filename = os.path.basename(file_path)

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [list(map(float, row)) for row in reader]  # Convert to float

    data = np.array(data)  # Convert to NumPy array

    # Extract AP values based on n
    ap_values = data[:, ::(n + 1)]  # Selecting every (n+1)th column

    # Compute average of AP values
    ap_averages = np.mean(ap_values, axis=1)
    final_throughput = 0.4 * (ap_averages / 500)

    # Append the result to the list
    throughput_results.append([final_throughput])

# Write the results to a new CSV file
output_file_path = "/home/gautam/Downloads/cappy-definitive-edition/step5/final_throughput.csv"  # Change to desired output path

with open(output_file_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["throughput"])  # Write header
    writer.writerows(throughput_results)  # Write the collected data

print(f"Final throughput values saved to {output_file_path}")
