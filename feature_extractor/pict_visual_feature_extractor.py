import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import h5py
import sys
import cv2
import pylab
import imageio
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

base_model = VGG19(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.get_layer("block5_pool").output) # vgg pool5 features

pict_dir = "picture/"
lis = os.listdir(pict_dir)
lis = sorted(lis)
len_data = len(lis)

print(lis[0])
print(len_data)
pict_features = np.zeros([len_data, 10, 7, 7, 512])

t = 10 # length of video
sample_num = 16 # frame number for each second

for num in range(len_data):
    #print("test")
    pict_index = os.path.join(pict_dir, lis[num]) #path of videos
    _image = cv2.imread(pict_index)
    _image = cv2.resize(_image, (224, 224))
    extract_frame = []
    #print("test2")
    for i in range(160):
        extract_frame.append(_image)
    feature = np.zeros(([10, 16, 7, 7, 512]))
    #print("test3")
    for j in range(len(extract_frame)):
        y_im = extract_frame[j]
        x = image.img_to_array(y_im)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pool_features = np.float32(model.predict(x))

        tt = int(j / sample_num)
        video_id = j - tt * sample_num
        feature[tt, video_id, :, :, :] = pool_features
        #print("test4")
    feature_vector = np.mean(feature, axis=(1)) # averaging features for 16 frames in each second
    pict_features[num, :, :, :, :] = feature_vector
    print(pict_features.shape)
    #np.save("features" + str(num) , feature_vector)

# save the visual features into one .h5 file. If you have a very large dataset, you may save each feature into one .npy file

with h5py.File('./pict_cnn_feature.h5', 'w') as hf:
    hf.create_dataset("dataset", data=pict_features)

