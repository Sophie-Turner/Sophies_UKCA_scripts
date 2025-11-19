from datetime import datetime, timedelta

suite = 'dt341'
year = 1982
start_date = datetime(year, 4, 15)
end_date = datetime(year + 1, 1, 1)
current_date = start_date

moo_get_cmds, rsync_cmds = [], []
while current_date < end_date:
  date = current_date.strftime('%m%d')
  filename = f'{suite}a.pl{year}{date}.pp'
  get = (f'moo get -v moose:/crum/u-{suite}/apl.pp/{filename} .')
  pull = (f'rsync -v sophiet@xfer-vm-01.jasmin.ac.uk:~/{filename} /scratch/st838/netscratch/data/ukca_pps/')
  moo_get_cmds.append(get)
  rsync_cmds.append(pull)
  current_date += timedelta(days=1)
  
max_files = 8  
  
print('rm rsync_done.flag')   
print('touch rsync_done.flag')  
for i in range(0, len(moo_get_cmds)-max_files, max_files):
  print('while [ ! -f rsync_done.flag ]; do')
  print('  sleep 300')
  print('done')
  print(f'rm {suite}a.* MetOffice* rsync_done.flag')
  for j in range(max_files):
    print(moo_get_cmds[i+j])
    
print()
 
print('rm rsync_done.flag')  
print('touch rsync_done.flag')   
for i in range(0, len(rsync_cmds)-max_files, max_files):
  print('sleep 3500')
  for j in range(max_files):
    print(rsync_cmds[i+j])
  print('rsync -v rsync_done.flag sophiet@xfer-vm-01.jasmin.ac.uk:~/')
