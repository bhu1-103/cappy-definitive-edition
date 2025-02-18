import os
import csv
import numpy as np
'''
input_folder_path = '/Users/gautammenon/Downloads/ohno/sec0a'
output_folder_path = '/Users/gautammenon/Downloads/ohno/performance_rssi'
'''
input_folder_path ='/home/gautam/Downloads/cappy-definitive-edition/step4/sce1a_output/rssi'
output_folder_path ='/home/gautam/Downloads/cappy-definitive-edition/step5/performance_rssi'
import os
import numpy as np
import csv

import os
import numpy as np
import csv
import glob

# User input for n
n = int(input("Enter the value of n: "))
# Specify the folder containing the CSV files
folder_path = "/home/gautam/Downloads/cappy-definitive-edition/step4/sce1a_output/rssi"  # Change this to your folder path

# Use glob to get all CSV files and sort them numerically based on filenames
csv_files = sorted(glob.glob(os.path.join(folder_path, "rssi_*.csv")), key=lambda x: int(x.split('_')[-1].split('.')[0]))

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

    print(f"Final RSSI for {filename}: {final_rssi}")

'''def calculate_segment_averages(values):
    segment_averages = []
    current_segment = []

    for value in values:
        if value == "Inf":
            if current_segment:
                segment_average = sum(current_segment) / len(current_segment)
                segment_averages.append(segment_average)
                current_segment = []
        else:
            current_segment.append(float(value))

    if current_segment:
        segment_average = sum(current_segment) / len(current_segment)
        segment_averages.append(segment_average)

    return segment_averages


def calculate_average_of_averages(segment_averages):
    if segment_averages:
        return sum(segment_averages) / len(segment_averages)
    return None


def calculate_all_averages_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        all_segment_averages = []

        for row in reader:
            values = row[0].split(',')
            segment_averages = calculate_segment_averages(values)
            all_segment_averages.extend(segment_averages)

        return calculate_average_of_averages(all_segment_averages)


def process_rssi_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for filename in os.listdir(input_folder):
        print(f"Found file: {filename}")
        if filename.startswith('rssi') and filename.endswith('.csv'):
            rssi_file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {rssi_file_path}")

            try:
                average_of_averages = calculate_all_averages_from_csv(rssi_file_path)
                if average_of_averages is not None:
                    output_file_path = os.path.join(output_folder,filename)
                    with open(output_file_path, 'w', newline='') as output_file:
                        csv_writer = csv.writer(output_file)
                        csv_writer.writerow([average_of_averages])
                    print(f"File written: {output_file_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Run the function
process_rssi_files(input_folder_path, output_folder_path)



# RSSI GOOD, AVERAGE AND BAD


import os
import csv

# Path to the new RSSI folder (input folder for this step)
#rssi_folder = '/Users/gautammenon/Downloads/ohno/performance_rssi'
rssi_folder ='/home/gautam/Downloads/cappy-definitive-edition/step5/performance_rssi'
# Ensure the performance_rssi directory exists
if not os.path.exists(rssi_folder):
    raise Exception("performance_rssi folder does not exist.")

# Initialize lists to store RSSI values for min/max calculation
rssi_values = []

# First pass: Read all files in the performance_rssi folder to collect RSSI values
for filename in os.listdir(rssi_folder):
    if filename.endswith('.csv'):
        input_file = os.path.join(rssi_folder, filename)

        with open(input_file, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                # Attempt to convert each value to float and add to the list
                for value in row:
                    try:
                        rssi_values.append(float(value))
                    except ValueError:
                        # Ignore non-numeric values
                        continue

# Calculate min and max RSSI if we have any values
if rssi_values:
    min_rssi = min(rssi_values)
    max_rssi = max(rssi_values)

    # Calculate range and define thresholds for good, average, and bad
    range_rssi = max_rssi - min_rssi
    good_threshold = min_rssi + range_rssi / 3
    average_threshold = min_rssi + 2 * range_rssi / 3

    # Second pass: Process each CSV file and tag them
    for filename in os.listdir(rssi_folder):
        if filename.endswith('.csv'):
            input_file = os.path.join(rssi_folder, filename)

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
                            # Tag each value based on its RSSI
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
                csv_writer.writerow(["Average RSSI", "RSSI Category"])
                # Write each tagged row
                for row in tagged_rows:
                    csv_writer.writerow(row)

            print(f"Processed {filename} and added tags to it.")

    print("All RSSI files have been processed.")
else:
    print("No valid RSSI values found in the performance_rssi folder.")
'''