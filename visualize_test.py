from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set GPU ID
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from dataloader import *
import random
import models_fusion
import time
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import imageio
import cv2
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

def video_frame_sample(frame_interval, video_length, sample_num):

    num = []
    for l in range(video_length):

        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))

    return num

# features, labels, and testing set list
dir_video  = "data/visual_feature.h5"
dir_audio  = "data/audio_feature.h5"
dir_labels = "data/labels.h5"
dir_order_test = "data/test_order.h5"

# access to original videos for extracting video frames
raw_video_dir = "data/AVE" # videos in AVE dataset
lis = os.listdir(raw_video_dir)
f = open("data/Annotations.txt", 'r')
dataset = f.readlines() 
print("The dataset contains %d samples" % (len(dataset)))
f.close()
len_data = len(dataset)
with h5py.File(dir_order_test, 'r') as hf:
    test_order = hf['order'][:]
    print(test_order.shape)

for num in range(len(test_order)):
    if num>=1:
        break
    print("num=",num)
    data = dataset[test_order[num]]
    print(data)
    x = data.split("&")
    print(x)

    video_index = os.path.join(raw_video_dir, x[1] + ".mp4")
    vid = imageio.get_reader(video_index, "ffmpeg")
    vid_len = len(vid)

    imgs = []
    for image in vid.iter_data():
        imgs.append(image)

    print(vid.get_meta_data())
    print(len(imgs)) #301
    vid.close()
    imgs = np.array(imgs)

    save_dir = "test/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i in range(len(imgs)):
        n = "%04d" % i
        plt.imsave(save_dir + str(n) + ".jpg", imgs[i])
