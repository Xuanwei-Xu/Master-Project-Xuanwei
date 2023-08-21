import glob
import shutil
import os

path_list = glob.glob("*/runs")
for path in path_list:
    print(path)
    shutil.rmtree(path)
print(path_list)
