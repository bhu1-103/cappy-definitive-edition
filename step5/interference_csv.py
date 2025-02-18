import os
import numpy as np
import csv
import glob

# User input for n
n = int(input("Enter the value of AP: "))

# Specify the folder containing the CSV files
folder_path = "/home/gautam/Downloads/cappy-definitive-edition/step4/sce1a_output/interference"  # Change this to your folder path

# Use glob to get all CSV files and sort them numerically based on filenames
csv_files = sorted(glob.glob(os.path.join(folder_path, "interference_*.csv")), key=lambda x: int(x.split('_')[-1].split('.')[0]))

# List to store the results for output CSV
interference_results = []

# Iterate through all CSV files in the folder
for file_path in csv_files:
    filename = os.path.basename(file_path)
    #print(f"Processing file: {filename}")

    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')  # Use correct delimiter
        data = []

        for row in reader:
            clean_row = []
            for cell in row:
                values = cell.split(',')  # Split multiple values in a cell
                for val in values:
                    val = val.strip()  # Remove extra spaces
                    if val.lower() == 'inf':
                        clean_row.append(np.nan)  # Replace 'inf' with NaN
                    elif val:
                        clean_row.append(float(val))
            if clean_row:  # Avoid adding empty rows
                data.append(clean_row)

    data = np.array(data)  # Convert to NumPy array

    # Ensure inf values along diagonal for n=3 case are replaced
    if data.shape[0] == data.shape[1]:  # Square matrix case (n=3)
        np.fill_diagonal(data, np.nan)

    # Compute average without considering NaN values
    overall_average = np.nanmean(data)  # nanmean ignores NaNs in mean calculation
    final_interference = 0.2 * (1 - ((overall_average + 100) / 100))

    # Append the result to the list
    interference_results.append([final_interference])

    #print(f"Final interference for {filename}: {final_interference}")

# Write the results to a new CSV file
output_file_path = "/home/gautam/Downloads/cappy-definitive-edition/step5/final_interference.csv"  # Change to desired output path

with open(output_file_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["interference"])  # Write header
    writer.writerows(interference_results)  # Write the collected data

print(f"Final interference values saved to {output_file_path}")
