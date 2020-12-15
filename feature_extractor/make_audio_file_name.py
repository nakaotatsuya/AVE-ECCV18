import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import h5py
import sys
import pylab
import pandas as pd
import glob

label_dir = "labels/"
lis = os.listdir(label_dir)

lis = sorted(lis)
print(lis[0])
len_data = len(lis)
print(len_data)

m = 0
r = 0
k = 0


flist_m = glob.glob("./audio/microwave/*.wav")
flist_r = glob.glob("./audio/fridge/*.wav")
flist_k = glob.glob("./audio/kettle/*.wav")
#print(flist_m)
print(len(flist_r))

for i in range(len_data):
    search_file = os.path.join(label_dir, lis[i])
    f = open(search_file)
    data = f.read().split()
    cate = data[0]
    print(cate)

    if cate == "microwave_oven":
        os.rename(flist_m[m], "./audio/" + lis[i].replace(".txt","") + ".wav")
        m+= 1
    elif cate == "refrigerator":
        os.rename(flist_r[r], "./audio/" + lis[i].replace(".txt","") + ".wav")
        r+= 1
    elif cate == "kettle":
        os.rename(flist_k[k], "./audio/" + lis[i].replace(".txt","") + ".wav")
        k+= 1
    f.close()

print(m)
print(r)
print(k)

