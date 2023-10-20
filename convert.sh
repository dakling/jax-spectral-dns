#!/usr/bin/env sh

find ./plots -type f -name "*.pdf" |while read line
do
   dir=${line%/*}
   file=${line##*/}
   file=${file%.*}
   convert -transparent white -fuzz 10% $line ${file}.png
   mv ${file}.png ${dir}/${file}.png
done
