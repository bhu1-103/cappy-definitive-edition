import os
import csv

# Function to read data from CSV files
def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first header row
        for row in reader:
            value = row[0].strip()  # Remove any leading/trailing whitespace
            if value.startswith("[") and value.endswith("]"):  # Check if value is in brackets
                value = value[1:-1]  # Remove brackets
            try:
                data.append(float(value))  # Convert to float
            except ValueError:
                print(f"Warning: Could not convert {row[0]} to float.")  # Handle any non-convertible values
    return data

# Specify the folder containing the CSV files
folder_path_throughput = "/home/gautam/Downloads/cappy-definitive-edition/step5"
folder_path_airtime = "/home/gautam/Downloads/cappy-definitive-edition/step5"
folder_path_rssi = "/home/gautam/Downloads/cappy-definitive-edition/step5"
folder_path_interference = "/home/gautam/Downloads/cappy-definitive-edition/step5"

# Assuming the filenames are consistent, get the first file from each folder
file_throughput = os.path.join(folder_path_throughput, "final_throughput.csv")
file_airtime = os.path.join(folder_path_airtime, "final_airtime.csv")
file_rssi = os.path.join(folder_path_rssi, "final_rssi.csv")
file_interference = os.path.join(folder_path_interference, "final_interference.csv")

# Read the data from each of the CSV files (assuming a single column in each)
throughput_data = read_data(file_throughput)
airtime_data = read_data(file_airtime)
rssi_data = read_data(file_rssi)
interference_data = read_data(file_interference)

# Combine the four data lists into one (zip them together to form rows)
combined_data = zip(throughput_data, airtime_data, rssi_data, interference_data)

# Write the combined data into a new CSV file with an additional column for performance score
output_file_path = "/home/gautam/Downloads/cappy-definitive-edition/step5/performance.csv"  # Change to desired output path

with open(output_file_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["throughput", "airtime", "rssi", "interference", "performance_score"])  # Write header
    for row in combined_data:
        performance_score = sum(row)  # Add the four columns to get the performance score
        writer.writerow(list(row) + [performance_score])  # Write the row with the performance score

print(f"Combined data with performance score saved to {output_file_path}")
