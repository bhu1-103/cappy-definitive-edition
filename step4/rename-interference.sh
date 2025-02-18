#!/bin/bash

DIRECTORY="sce1a_output/interference"
if [ ! -d "$DIRECTORY" ]; then
    echo "Directory $DIRECTORY not found!"
    exit 1
fi

cd "$DIRECTORY" || exit

for file in interference_*.csv; do
    number=$(echo "$file" | sed 's/interference_\([0-9]*\)\.csv/\1/')
    new_number=$(printf "%03d" "$((number - 1))")
    new_file="interference_${new_number}.csv"
    mv "$file" "$new_file"
done

