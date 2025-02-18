# AVERAGE AIRTIME


import csv
import os
import numpy as np
'''
# Input and output directories
input_folder_path = '/Users/gautammenon/Downloads/ohno/sec0a'
output_folder_path = '/Users/gautammenon/Downloads/ohno/performance_airtime'
'''
input_folder_path = '/home/gautam/Downloads/cappy-definitive-edition/step4/sce1a_output/airtime'
output_folder_path ='/home/gautam/Downloads/cappy-definitive-edition/step5/performance_airtime'
# Create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

import os
import numpy as np
import csv
import glob

# Specify the folder containing the CSV files
folder_path = "/home/gautam/Downloads/cappy-definitive-edition/step4/sce1a_output/airtime"  # Change this to your folder path

# Use glob to get all CSV files and sort them numerically based on filenames
csv_files = sorted(glob.glob(os.path.join(folder_path, "airtime_*.csv")), key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Iterate through all CSV files in the folder
for file_path in csv_files:
    filename = os.path.basename(file_path)
    #print(f"Processing file: {filename}")

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

    #print(f"Overall average value for {filename}: {overall_average}")
    print(f"Final airtime for {filename}: {final_airtime}")

'''
# Function to calculate the average of averages for a file
def calculate_average_of_averages(file_path):
    averages = []

    # Read the CSV file
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            for range_str in row:
                # Step 1: Split the range by semicolons
                ranges = range_str.split(';')

                # Step 2: Process each range
                for range_item in ranges:
                    if range_item.strip():
                        # Split the range by commas and convert to float
                        values = list(map(float, range_item.split(',')))

                        # Calculate the average for the current range
                        if values:
                            avg = sum(values) / len(values)
                            averages.append(avg)

    # Step 3: Calculate the average of averages
    if averages:
        return sum(averages) / len(averages)
    else:
        return None


# Loop through all files in the input folder
for filename in os.listdir(input_folder_path):
    if filename.endswith('.csv') and filename.startswith('airtime'):  # Only process CSV files
        input_file_path = os.path.join(input_folder_path, filename)
        average_of_averages = calculate_average_of_averages(input_file_path)

        # If a valid average is found, write it to a new file in the output folder
        if average_of_averages is not None:
            output_file_path = os.path.join(output_folder_path, f'{filename}')
            with open(output_file_path, mode='w', newline='') as output_file:
                csv_writer = csv.writer(output_file)

                csv_writer.writerow([average_of_averages])

print("Processing complete. Results are saved in the performance_airtime folder.")

# AIRTIME GOOD, BAD AND AVERAGE


import os
import csv

# Path to the performance_airtime folder (input folder for this step)
#airtime_folder = '/Users/gautammenon/Downloads/ohno/performance_airtime'
airtime_folder ='/home/gautam/Downloads/cappy-definitive-edition/step5/performance_airtime'
# Ensure the performance_airtime directory exists
if not os.path.exists(airtime_folder):
    raise Exception("performance_airtime folder does not exist.")

with open(file_path, 'r') as file:
    reader = csv.reader(file, delimiter=';')  # Use correct delimiter
    data = []

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

print("Overall average value:", overall_average)

# Initialize lists to store airtime values for min/max calculation
airtime_values = []

# First pass: Read all files in the performance_airtime folder to collect airtime values
for filename in os.listdir(airtime_folder):
    if filename.endswith('.csv'):
        input_file = os.path.join(airtime_folder, filename)

        with open(input_file, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                # Attempt to convert each value to float and add to the list
                for value in row:
                    try:
                        airtime_values.append(float(value))
                    except ValueError:
                        # Ignore non-numeric values
                        continue

# Calculate min and max airtime if we have any values
if airtime_values:
    min_airtime = min(airtime_values)
    max_airtime = max(airtime_values)

    # Calculate range and define thresholds for good, average, and bad
    range_airtime = max_airtime - min_airtime
    good_threshold = min_airtime + range_airtime / 3
    average_threshold = min_airtime + 2 * range_airtime / 3

    # Second pass: Process each CSV file and tag them
    for filename in os.listdir(airtime_folder):
        if filename.endswith('.csv'):
            input_file = os.path.join(airtime_folder, filename)

            # Initialize a list to hold tagged values
            tagged_rows = []

            # Read the CSV file and tag the floating-point values in the row
            with open(input_file, mode='r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    tagged_row = []
                    for value in row:
                        try:
                            float_value = float(value)
                            # Tag each value based on its airtime
                            if float_value < good_threshold:
                                tag = "bad"
                            elif float_value < average_threshold:
                                tag = "average"
                            else:
                                tag = "good"
                            tagged_row.append(float_value)
                            tagged_row.append(tag)  # Append the tag
                        except ValueError:
                            # Append non-numeric values unchanged
                            tagged_row.append(value)
                    tagged_rows.append(tagged_row)

            # Write the tagged rows to the corresponding output CSV file (overwriting the original)
            with open(input_file, mode='w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(["Average airtime", "airtime Category"])
                # Write each tagged row
                for row in tagged_rows:
                    csv_writer.writerow(row)

            print(f"Processed {filename} and added tags to it.")

    print("All airtime files have been processed.")
else:
    print("No valid airtime values found in the performance_airtime folder.")
'''