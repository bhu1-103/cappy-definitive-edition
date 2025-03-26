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
            number = "";
        }
        print prefix, number, $2, $4, $5, $6
    }' OFS=";" "$file" > temp && mv temp "$file"
    cat $file | awk -F ";" '{print $1,$3,$4,$5,$6}' OFS=";" > temp && mv temp $file
done
