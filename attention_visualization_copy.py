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

scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
def video_frame_sample(frame_interval, video_length, sample_num):

    num = []
    for l in range(video_length):

        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))

    return num

def normlize(x, min = 0, max = 255):

    num, row, col = x.shape
    for i in range(num):
        xi = x[i, :, :]
        xi = max *(xi - np.min(xi))/(np.max(xi) - np.min(xi))
        x[i, :, :] = xi
    return x

def create_heatmap(im_map, im_cloud, kernel_size=(5,5),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.3):
    print(np.max(im_cloud))

    im_cloud[:, :, 1] = 0
    im_cloud[:, :, 2] = 0
    return (a1*im_map + a2*im_cloud).astype(np.uint8)


# features, labels, and testing set list
dir_video  = "feature_extractor/pict_cnn_feature.h5"
dir_audio  = "feature_extractor/audio_embedding.h5"
dir_labels = "feature_extractor/labels.h5"
dir_order_test = "feature_extractor/test_order.h5"

# access to original videos for extracting video frames
raw_video_dir = "feature_extractor/picture" # videos in AVE dataset
raw_label_dir = "feature_extractor/labels"
#lis = os.listdir(raw_video_dir)
lis = os.listdir(raw_video_dir)
lis = sorted(lis)
print(lis)
#f = open("data/Annotations.txt", 'r')
#dataset = f.readlines() 
#print("The dataset contains %d samples" % (len(dataset)))
#f.close()
#len_data = len(dataset)

with h5py.File(dir_order_test, 'r') as hf:
    test_order = hf['order'][:]
    #print(test_order)

# pre-trained models
att_model = torch.load('model/AV_att.pt')
att_layer = att_model._modules.get('affine_h') # extract attention maps from the layer


# load testing set
AVEData = Dataset(video_dir=dir_video, audio_dir=dir_audio, label_dir=dir_labels,order_dir=dir_order_test, batch_size=57)
nb_batch = AVEData.__len__()
print(nb_batch) #57
audio_inputs, video_inputs, labels = AVEData.get_batch(0)
audio_inputs = Variable(audio_inputs.cuda(), requires_grad=False)
video_inputs = Variable(video_inputs.cuda(), requires_grad=False)
print(video_inputs.shape) #(402,10,7,7,512)
labels = labels.numpy()
print(labels.shape) #(402,10,29)

# generate attention maps

att_map = torch.zeros((570, 49, 1))

def fun(m, i, o): att_map.copy_(o.data)
map = att_layer.register_forward_hook(fun)
h_x = att_model(audio_inputs, video_inputs)
print(h_x.shape) #(402,10,29)
map.remove()
z_t = Variable(att_map.squeeze( 2 ))
alpha_t = F.softmax( z_t, dim = -1 ).view( z_t.size( 0 ), -1, z_t.size( 1 ) )
att_weight = alpha_t.view(57, 10, 7, 7).cpu().data.numpy() # attention maps of all testing samples
    
c = 0
t = 10
sample_num = 16 # 16 frames for 1-sec video segment
extract_frames = np.zeros((160, 224, 224, 3)) # 160 224x224x3 frames for a 10-sec video
save_dir = "visual_att/my_attention_maps/" # store attention maps
original_dir = "visual_att/my_original/"   # store video frames

for num in range(len(test_order)):
    print("num=", num)
    #print(len(test_order)) #402
    #print(test_order)
    
    data = lis[test_order[num]]
    print(data)
    #x = data.split("&")
    #print(x)
    
    # extract video frames
    pict_index = os.path.join(raw_video_dir, data)

    #vid = imageio.get_reader(video_index, 'ffmpeg')
    _image = cv2.imread(pict_index)
    _image = cv2.resize(_image, (224,224))
    print(_image.shape)
    #vid_len = len(vid)
    #print(vid_len)
    #frame_interval = int(vid_len / t)
    #print(frame_interval)
    
    imgs = []
    for i in range(160):
        imgs.append(_image)
    #for i, im in enumerate(vid):
    #    x_im = cv2.resize(im, (224, 224))
    #    imgs.append(x_im)
    print("imgs =",len(imgs))

    #frame_interval = int(len(imgs) / t)
    #frame_num = video_frame_sample(frame_interval, t, sample_num)
    #print(len(frame_num)) #160
    #vid.close()
    
    #cc = 0
    #for n in frame_num:
    #    #print(imgs[160])
    #    #print(n)
    #    extract_frames[cc, :, :, :] = imgs[n]
    #    cc += 1

    for i in range(160):
        extract_frames[i, :, :, :] = imgs[i]

    print(extract_frames.shape) #(160,224,224,3)
    # process generated attention maps
    att = att_weight[num, :, :, :]
    att = normlize(att, 0, 255)
    print(att.shape) #10,7,7
    att_scaled = np.zeros((10, 224, 224))
    for k in range(att.shape[0]):
        att_scaled[k, :, :] = cv2.resize(att[k, :, :], (224, 224)) # scaling attention maps
        
  
    att_t = np.repeat(att_scaled, 16, axis = 0) # 1-sec segment only has 1 attention map. Here, repeat 16 times to generate 16 maps for a 1-sec video
    heat_maps = np.repeat(att_t.reshape(160, 224, 224, 1), 3, axis = -1)
    c += 1
    
    att_dir = save_dir + data.replace(".jpg", "")
    ori_dir =  original_dir + data.replace(".jpg", "")
    if not os.path.exists(att_dir):
      os.makedirs(att_dir)
    if not os.path.exists(ori_dir):
      os.makedirs(ori_dir)
    for idx in range(160):
        heat_map = heat_maps[idx, :, :, 0]
        im = extract_frames[idx, :, :, :]
        im = im[:, :, (2, 1, 0)]
        heatmap = cv2.applyColorMap(np.uint8(heat_map), cv2.COLORMAP_JET)
        
        att_frame = heatmap * 0.2 + np.uint8(im) * 0.6
        n = "%04d" % idx
        vid_index = os.path.join(att_dir, 'pic' + n + '.jpg')
        cv2.imwrite(vid_index, att_frame)
        ori_frame = np.uint8(im)
        ori_index = os.path.join(ori_dir, 'ori' + n + '.jpg')
        cv2.imwrite(ori_index, ori_frame)

