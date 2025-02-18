#AVERAGE INTERFERENCE
import csv
import os
import numpy as np



'''# Define the folder paths
input_folder = r'/Users/gautammenon/Downloads/ohno/sec0a'
output_folder = r'/Users/gautammenon/Downloads/ohno/performance_interference'
'''
input_folder = '/home/gautam/Downloads/cappy-definitive-edition/step4/sce1a_output/interference'
output_folder ='/home/gautam/Downloads/cappy-definitive-edition/step5/performance_interference'
# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

import os
import numpy as np
import csv
import os
import numpy as np
import csv
import glob
import os
import numpy as np
import csv
import glob

# User input for n
n = int(input("Enter the value of n: "))

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

    print(f"Final interference for {filename}: {final_interference}")

# Write the results to a new CSV file
output_file_path = "/home/gautam/Downloads/cappy-definitive-edition/step4/sce1a_output/final_interference.csv"  # Change to desired output path

with open(output_file_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["interference"])  # Write header
    writer.writerows(interference_results)  # Write the collected data

print(f"Final interference values saved to {output_file_path}")

'''
# Iterate through all the CSV files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv") and filename.startswith('interference'):
        file_path = os.path.join(input_folder, filename)

        # Initialize total sum for the current file
        total_sum = 0.0

        # Open and read the CSV file
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                # Split each value in the row by commas and iterate over the values
                for value in row[0].split(','):
                    if value != "Inf":  # Ignore 'Inf'
                        total_sum += float(value)  # Convert to float and add to total sum

        # Calculate the average by dividing the sum by n^2
        average = total_sum / (n ** 2)

        # Define the output file path
        output_file_path = os.path.join(output_folder, filename)

        # Write the result into the output CSV file
        with open(output_file_path, 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            writer.writerow([average])  # Write the average value in the first row

print(f"All averages have been saved to the 'performance_interference' folder.")






# INTERFERENCE GOOD, BAD AND AVERAGE

import os
import csv

# Path to the performance_interference folder (input folder for this step)
#interference_folder = '/Users/gautammenon/Downloads/ohno/performance_interference'
interference_folder ='/home/gautam/Downloads/cappy-definitive-edition/step5/performance_interference'
# Ensure the performance_interference directory exists
if not os.path.exists(interference_folder):
    raise Exception("performance_interference folder does not exist.")

# Initialize a list to store interference values for min/max calculation
interference_values = []

# First pass: Read all files in the performance_interference folder to collect interference values
for filename in os.listdir(interference_folder):
    if filename.endswith('.csv'):
        input_file = os.path.join(interference_folder, filename)

        with open(input_file, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                # Attempt to convert each value to float and add to the list
                for value in row:
                    try:
                        interference_values.append(float(value))
                    except ValueError:
                        # Ignore non-numeric values
                        continue

# Calculate min and max interference if we have any values
if interference_values:
    min_interference = min(interference_values)
    max_interference = max(interference_values)

    # Calculate range and define thresholds for good, average, and bad
    range_interference = max_interference - min_interference
    good_threshold = min_interference + range_interference / 3
    average_threshold = min_interference + 2 * range_interference / 3

    # Second pass: Process each CSV file and tag them
    for filename in os.listdir(interference_folder):
        if filename.endswith('.csv') and filename.startswith('interference'):
            input_file = os.path.join(interference_folder, filename)

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
                            # Tag each value based on its interference level
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
                csv_writer.writerow(["Average interference", "interference Category"])
                # Write each tagged row
                for row in tagged_rows:
                    csv_writer.writerow(row)

            print(f"Processed {filename} and added tags to it.")

    print("All interference files have been processed.")
else:
    print("No valid interference values found in the performance_interference folder.")'''