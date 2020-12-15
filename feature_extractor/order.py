import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import h5py
import sys
import pylab
import random
#order = [0,1]

pict_dir = "picture/"
lis = os.listdir(pict_dir)
lis = sorted(lis)
len_data = len(lis)

lr = list(range(len_data))
lr = random.sample(lr, len(lr))
print(lr)

# l = list(range(5))
# print(l)
# random.shuffle(l)
# print(l)

#train_data = len_data * 0.8
val_data = len_data // 10 
test_data = len_data // 10
train_data = len_data - (val_data + test_data)

train_order = lr[:train_data]
val_order = lr[train_data:(train_data+val_data)]
test_order = lr[(train_data+val_data):]

#print(val_order)

with h5py.File("./train_order.h5", "w") as hf:
    hf.create_dataset("order", data=train_order)

with h5py.File("./val_order.h5", "w") as hf:
    hf.create_dataset("order", data=val_order)

with h5py.File("./test_order.h5", "w") as hf:
    hf.create_dataset("order", data=test_order)
