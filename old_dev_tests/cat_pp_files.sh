#!/bin/bash
# Stick output files together.

# Loop through all .pl files
for plfile in dv856a.pl*.pp; do
    # Extract date from filename: characters after 'pl' and before '.pp'
    date=${plfile#dv856a.pl}
    date=${date%.pp}
    echo ${date}
    # Construct matching pn filename
    pnfile="dv856a.pn${date}.pp"
    pqfile="dv856a.pq${date}.pp"
    newfile="dv856a.px${date}.pp"
    # Concatenate files.
    cat "$pnfile" "$plfile" "$pqfile" > "$newfile"
done
