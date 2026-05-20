'''Check for missing files in monthly datasets.'''

import os
import glob
import file_paths as paths

suite = 'dw370'
dir_path = f'{paths.data}/ml30yr_masked'
files_npy = glob.glob(f'{dir_path}/*.npy')
files_pp = glob.glob(f'{dir_path}/{suite}a.px*.pp')
run_start = 1982 # Start year in run.
run_len = 30 # Num years run.
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
years = range(run_start, run_start + run_len)

count_npy, count_pp = 0, 0
for year in years:
  for month in months:
    date = str(year) + month
    filename_npy = f'{dir_path}/{date}.npy'
    filename_pp = f'{dir_path}/{suite}a.px{date}.pp'
    if filename_npy not in files_npy:
      count_npy += 1
      if filename_pp not in files_pp:
        count_pp +=1
        print(date, 'is missing from both datasets')
      else:
        print(date, 'is missing from the npy dataset but is present in the pp dataset.')
print(count_npy, 'files are missing from the npy data.')
print(count_pp, 'files are missing from the pp data.')
