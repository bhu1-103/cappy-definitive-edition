import os
import glob

def fix_delimiters_in_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    if ';' in content:
        content = content.replace(';', ',')
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"✔️ Fixed delimiters in: {file_path}")
    else:
        print(f"✅ Already clean: {file_path}")

# Target root folder
base_path = "step6/100-files"

# Fix step2 and step4 files
for subfolder in glob.glob(f"{base_path}/*"):
    z_output_path = os.path.join(subfolder, "step2", "z_output")
    sce1a_output_path = os.path.join(subfolder, "step4", "sce1a_output")

    for path in [z_output_path, sce1a_output_path]:
        if not os.path.isdir(path):
            continue
        for csv_file in glob.glob(f"{path}/*.csv"):
            fix_delimiters_in_file(csv_file)
