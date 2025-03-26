#!/usr/bin/zsh
for file in in/*.csv
do
	cat $file | awk -F ";" '{print $1,$4,$5,$6}' OFS=";" > temp && mv temp $file
	#cat $file | awk -F " " '{print $1";"$2";"$3";"$4}' > temp && mv temp $file
done
