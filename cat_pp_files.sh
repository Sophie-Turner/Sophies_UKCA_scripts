#!/bin/bash
# Stick output files together.

# Loop through all .pl files
for plfile in dt341a.pl*.pp; do
    # Extract date from filename: characters after 'pl' and before '.pp'
    date=${plfile#dt341a.pl}
    date=${date%.pp}
    # Construct matching pn filename
    pnfile="dt341a.pn${date}.pp"
    pqfile="dt341a.pq${date}.pp"
    newfile="dt341a.px${date}.pp"
    # Concatenate files.
    cat "$pnfile" "$plfile" "$pqfile" > "$newfile"
done
