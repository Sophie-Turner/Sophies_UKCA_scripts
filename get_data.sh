#!/bin/bash
# Run from atmos server.
# Run at same time as its counterpart on MASS server.

SUITE="dv856"
DIR="ml1yr"

# Make a ready flag.
touch "done.flag"
# Send the ready flag to the server to start things off.
rsync "done.flag" "sophiet@xfer-vm-01.jasmin.ac.uk:~/"

# For every output stream and month...
for STREAM in {01..03}; do
  for MONTH in {01..12}; do   
  
    # Wait for the data to be moo fetched by the MASS server.
     sleep 1250 # 20 mins.
  
    # Fetch all the relevant data on the server.
     rsync -av --ignore-existing --partial --partial-dir=.rsync-partial --info=progress2 "sophiet@xfer-vm-01.jasmin.ac.uk:~/${SUITE}*" "/scratch/st838/netscratch/data/${DIR}/"
  
    # Send the ready flag to the server.
    rsync "done.flag" "sophiet@xfer-vm-01.jasmin.ac.uk:~/"
  
  done
done

rm "done.flag"
