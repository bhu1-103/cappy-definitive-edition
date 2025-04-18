#!/usr/bin/zsh

rm -rf in; cp -r ../step2/z_output in
rm -rf out; cp -r ../step4/sce1a_output/throughput out

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
    cat $file | awk -F ";" '{printf "%s;%d;%0.2f;%0.2f;%0.2f\n",$1,$3,$4,$5,$6}' OFS=";" > temp && mv temp $file
    sed -i '1d' $file
done

for file in out/*.csv
do
	tr -s ','  '\n'< $file > temp && mv temp $file
done
