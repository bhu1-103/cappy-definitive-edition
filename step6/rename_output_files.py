
import os
import glob
import re

def natural_sort_key(s):
    """Key for natural sorting (e.g., 1, 2, 10 instead of 1, 10, 2)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def rename_throughput_files(base_dir):
    """Renames throughput CSV files to align with input file numbering (000-indexed)."""
    all_subdirs = sorted(glob.glob(os.path.join(base_dir, '*')), key=natural_sort_key)

    for subdir in all_subdirs:
        throughput_dir = os.path.join(subdir, 'step4/sce1a_output/')
        if not os.path.exists(throughput_dir):
            print(f"Throughput directory not found: {throughput_dir}. Skipping.")
            continue

        throughput_files = sorted(glob.glob(os.path.join(throughput_dir, 'throughput_*.csv')), key=natural_sort_key)

        for old_path in throughput_files:
            filename = os.path.basename(old_path)
            match = re.search(r'throughput_(\d+)\.csv', filename)
            if match:
                original_num = int(match.group(1))
                # Subtract 1 to make it 0-indexed, then format with leading zeros
                new_num = original_num - 1
                if new_num < 0:
                    print(f"Warning: Skipping {filename} as its index would be negative after adjustment.")
                    continue
                
                # Format to 3 digits (e.g., 0 -> 000, 9 -> 009, 99 -> 099)
                new_filename = f'throughput_{new_num:03d}.csv'
                new_path = os.path.join(throughput_dir, new_filename)

                if old_path != new_path:
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed: {os.path.basename(old_path)} -> {new_filename}")
                    except OSError as e:
                        print(f"Error renaming {os.path.basename(old_path)} to {new_filename}: {e}")
            else:
                print(f"Warning: Could not parse number from {filename}. Skipping.")

if __name__ == '__main__':
    data_base_directory = '/home/bhu1/dev/git/cappy-definitive-edition/step6/100-files/'
    print(f"Starting file renaming in {data_base_directory}...")
    rename_throughput_files(data_base_directory)
    print("File renaming complete.")
