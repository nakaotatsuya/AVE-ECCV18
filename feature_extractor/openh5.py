import numpy as np
import h5py

# with h5py.File("video_cnn_feature.h5", mode="r") as f:
#     print("--------------")
#     for k in f:
#         print(k)
#     print("--------------")
#     count = 0
#     for k in f["dataset"]:
#         count +=1
#         print(k.shape)
#     print(count)

with h5py.File("labels.h5", mode="r") as f:
    print("--------------")
    for k in f:
        print(k)
    print("--------------")
    count = 0
    for k in f["dataset"]:
        count +=1
        print(k)
    print(count)

# with h5py.File("order_test.h5", mode="r") as f:
#     print("--------------")
#     for k in f:
#         print(k)
#     print("--------------")
#     count = 0
#     for k in f["order"]:
#         count +=1
#         print(k)
#     print(count)

# with h5py.File("pict_cnn_feature.h5", mode="r") as f:
#     print("--------------")
#     for k in f:
#         print(k)
#     print("--------------")
#     count = 0
#     for k in f["dataset"]:
#         count +=1
#         print(k.shape)
#         #print(k)
#     print(count)
