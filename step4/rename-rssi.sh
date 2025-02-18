#!/bin/bash

DIRECTORY="sce1a_output/rssi"
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory $DIRECTORY not found!"
    exit 1
fi

cd "$DIRECTORY" || exit

for file in rssi_*.csv; do
    number=$(echo "$file" | sed 's/rssi_\([0-9]*\)\.csv/\1/')
    new_number=$(printf "%03d" "$((number - 1))")
    new_file="rssi_${new_number}.csv"
    mv "$file" "$new_file"
done

