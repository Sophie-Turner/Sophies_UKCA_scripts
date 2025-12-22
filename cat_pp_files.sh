#!/bin/bash

# Loop through all .pl files
for plfile in dt341a.pl*.pp; do
    # Extract date from filename: characters after 'pl' and before '.pp'
    date=${plfile#dt341a.pl}
    date=${date%.pp}
    # Construct matching pn filename
    pnfile="dt341a.pn${date}.pp"
    newfile="dt341a.px${date}.pp"
    # Concatenate files.
    cat "$plfile" "$pnfile" > "$newfile"
done
