import os
import csv
import numpy as np
# Get the value of k from the terminal
#k = int(input("Enter the value of k: "))
'''
# Path to the throughput folder
input_folder = '/Users/gautammenon/Downloads/ohno/sec0a'

# Path to the performance folder (output folder)
output_folder = '/Users/gautammenon/Downloads/ohno/performance'

'''
input_folder = '/home/gautam/Downloads/cappy-definitive-edition/step4/sce1a_output/throughput'
output_folder ='/home/gautam/Downloads/cappy-definitive-edition/step5/performance'
# Ensure the output directory exists, create it if necessary
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

import os
import numpy as np
import csv
import glob

# User input for n
n = int(input("Enter the value of n: "))

# Specify the folder containing the CSV files
folder_path = "/home/gautam/Downloads/cappy-definitive-edition/step4/sce1a_output/throughput"  # Change this to your folder path

# Use glob to get all CSV files and sort them numerically based on filenames
csv_files = sorted(glob.glob(os.path.join(folder_path, "throughput_*.csv")), key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Iterate through all CSV files in the folder
for file_path in csv_files:
    filename = os.path.basename(file_path)
    #print(f"Processing file: {filename}")

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [list(map(float, row)) for row in reader]  # Convert to float

    data = np.array(data)  # Convert to NumPy array

    # Extract AP values based on n
    ap_values = data[:, ::(n + 1)]  # Selecting every (n+1)th column

    # Compute average of AP values
    ap_averages = np.mean(ap_values, axis=1)
    final_throughput = 0.4 * (ap_averages / 500)

    # Print final throughput for each file
    print(f"Final throughput for {filename}: {final_throughput}")


'''
# Initialize a counter for naming the performance files
file_counter = 1

# Process each CSV file in the throughput folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv') and filename.startswith('throughput'):
        input_file = os.path.join(input_folder, filename)

        # Create the output file name dynamically
        output_file = os.path.join(output_folder, f'performance_{file_counter}.csv')

        # Initialize the sum of the row
        total_sum = 0

        # Read the CSV file and sum the floating-point values in the row
        with open(input_file, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                # Convert all values in the row to floats and sum them
                total_sum = sum(map(float, row))

        # Calculate the average by dividing the total sum by k
        average = total_sum / k

        # Write the average value to the corresponding output CSV file
        with open(output_file, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            # Write the header first

            # Write the average value
            csv_writer.writerow([average])

        print(f"Processed {filename} and written to {output_file}")

        # Increment the file counter for the next file
        file_counter += 1
import os
import csv

# Path to the performance folder (input folder for this step)
#performance_folder = '/Users/gautammenon/Downloads/ohno/performance'
performance_folder ='/home/gautam/Downloads/cappy-definitive-edition/step5/performance'
# Ensure the performance directory exists
if not os.path.exists(performance_folder):
    raise Exception("Performance folder does not exist.")

# Initialize lists to store throughput values for min/max calculation
throughput_values = []

# First pass: Read all files in the performance folder to collect throughput values
for filename in os.listdir(performance_folder):
    if filename.endswith('.csv'):
        input_file = os.path.join(performance_folder, filename)

        with open(input_file, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                # Attempt to convert each value to float and add to the list
                for value in row:
                    try:
                        throughput_values.append(float(value))
                    except ValueError:
                        # Ignore non-numeric values
                        continue

# Calculate min and max throughput if we have any values
if throughput_values:
    min_throughput = min(throughput_values)
    max_throughput = max(throughput_values)

    # Calculate range and define thresholds for good, average, and bad
    range_throughput = max_throughput - min_throughput
    good_threshold = min_throughput + range_throughput / 3
    average_threshold = min_throughput + 2 * range_throughput / 3

    # Second pass: Process each CSV file and tag them
    for filename in os.listdir(performance_folder):
        if filename.endswith('.csv'):
            input_file = os.path.join(performance_folder, filename)

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
                            # Tag each value based on its throughput
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

                # Add the header row with "Average Throughput" and "Throughput Category"
                csv_writer.writerow(["Average Throughput", "Throughput Category"])

                # Write each tagged row
                for row in tagged_rows:
                    # Get the last value from the tagged row (throughput value)
                    throughput_value = row[0] if row else 0
                    # Write the average throughput and its category
                    csv_writer.writerow([throughput_value, row[1] if len(row) > 1 else ""])  # Add category if it exists

            print(f"Processed {filename} and added tags to it.")

    print("All files have been processed.")
else:
    print("No valid throughput values found in the performance folder.")
'''