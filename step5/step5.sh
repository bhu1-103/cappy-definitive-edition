#!/bin/zsh

python3 throughput_csv.py

python3 interference_csv.py

python3 airtime_csv.py

python3 rssi_csv.py
 
python3 final.py


# Path to the CSV file
csv_file="/home/gautam/Downloads/cappy-definitive-edition/step5/performance.csv"

# Open the CSV file in LibreOffice Calc
libreoffice --calc "$csv_file"
