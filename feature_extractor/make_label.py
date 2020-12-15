import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import h5py
import sys
import pylab
import pandas as pd

label_dir = "labels/"
#pict_dir = "picture/"
lis = os.listdir(label_dir)
#lis_pic = os.listdir(pict_dir)

lis = sorted(lis)
#lis_pic = sorted(lis_pic)
print(lis[0])
#print(lis_pic[232])
len_data = len(lis)
print(len_data)
labels = np.zeros([len_data, 10, 4]) #29 to 4 (now 4 categories)
#29 categories (28 + 1 background)


#data = pd.read_table(search_file, header=None)
#print(data[0])

m = 0
r = 0
k = 0
for i in range(len_data):
    search_file = os.path.join(label_dir, lis[i])
    f = open(search_file)
    data = f.read().split()
    cate = data[0]
    print(cate)

    for j in range(10):
        if cate == "microwave_oven":
            labels[i][j][0] = 1
            m+= 1
        elif cate == "refrigerator":
            labels[i][j][1] = 1
            r+= 1
        elif cate == "kettle":
            labels[i][j][2] = 1
            k+= 1
    f.close()

print(m)
print(r)
print(k)

# for i in range(10):
#     labels[0][i][4] = 1
#     labels[1][i][1] = 1

#labels[1][0:10][1] = 1

# with h5py.File("./labels.h5", "w") as hf:
#     hf.create_dataset("dataset", data=labels)
