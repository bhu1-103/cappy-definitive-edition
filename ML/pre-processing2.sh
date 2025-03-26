#!/usr/bin/zsh
for file in in/*.csv
do
    awk -F ";" '{
        gsub(/^STA_|^AP_/, "", $1);  # Remove STA_ and AP_
        match($1, /[0-9]/);
        if (RSTART > 0) {
            prefix = substr($1, 1, RSTART - 1);
            number = substr($1, RSTART);
        } else {
            prefix = $1;
            number = "0";  # Assign 0 explicitly for APs
        }
        print prefix, number, ($number == "0" ? "0" : "1"), $2, $3, $4
    }' OFS=";" "$file" > tmpfile && mv tmpfile "$file"
done
