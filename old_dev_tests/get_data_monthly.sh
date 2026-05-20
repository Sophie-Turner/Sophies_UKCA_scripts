#!/bin/bash
# Run from atmos server.
# Run at same time as its counterpart on MASS server.

SUITE="dw370"
DIR="ml30yr_masked"

# Make a ready flag.
touch "done.flag"
# Send the ready flag to the server to start things off.
rsync "done.flag" "sophiet@xfer-vm-01.jasmin.ac.uk:~/"

# For every output stream and year...
for YEAR in {1988..2011}; do    
  
  # Wait for the data to be moo fetched by the MASS server.
  echo "Waiting 30 minutes for server to get data from MASS."
  sleep 1800 # 30 mins.
  
  # Fetch all the relevant data on the server.
  echo "Fetching year ${YEAR}." 
  rsync -av --ignore-existing --partial --partial-dir=.rsync-partial --info=progress2 "sophiet@xfer-vm-01.jasmin.ac.uk:~/${SUITE}*" "/scratch/st838/netscratch/data/${DIR}/"
  
  # Send the ready flag to the server.
  echo "Sending ready flag to server to continue."
  rsync "done.flag" "sophiet@xfer-vm-01.jasmin.ac.uk:~/"
  
done

rm "done.flag"
