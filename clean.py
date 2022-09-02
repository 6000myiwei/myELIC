#%%
import glob
import shutil
import os

log_files = glob.glob('pretrained/**/*.log', recursive=True)
for log in log_files:
    with open(log, 'r') as f:
        lines = f.readlines()
    if len(lines) <= 50:
        os.remove(log)
#%%