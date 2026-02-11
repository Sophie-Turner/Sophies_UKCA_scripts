#!/bin/bash
# Stick output files together.

# Loop through all .pl files
for plfile in dw370a.pl*.pp; do
    # Extract date from filename: characters after 'pl' and before '.pp'
    date=${plfile#dw370a.pl}
    date=${date%.pp}
    echo ${date}
    # Construct matching pn filename
    pnfile="dw370a.pn${date}.pp"
    pqfile="dw370a.pq${date}.pp"
    newfile="dw370a.px${date}.pp"
    # Concatenate files.
    cat "$pnfile" "$plfile" "$pqfile" > "$newfile"
done
