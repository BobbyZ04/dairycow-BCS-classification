# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:09:12 2021

@author: BOZ
"""

from tqdm import tqdm
from skimage.io import imread
from skimage.transform import rotate,resize
from skimage.util import random_noise
import numpy as np
from sklearn.model_selection import train_test_split
import torch

# load images
def load_dataset(img_names,data_path,labels):
    
    train_img = []
    for img_name in tqdm(img_names):
        img_path =  data_path + img_name
        img = imread(img_path)
        img_resized = resize(img,(224,224),anti_aliasing=True)
        train_img.append(img_resized)

    train_x = np.array(train_img)
    train_y = labels
    print('training set: ',train_x.shape, '\t trainng label: ', train_y.shape)
    train_x, val_x, train_y, val_y = train_test_split(train_x,train_y,test_size = 0.2,random_state=42, stratify=train_y)
    print('training set: ', (train_x.shape,train_y.shape),'\t testing set: ', (val_x.shape,val_y.shape))
    return train_x, val_x, train_y, val_y

# augment dataset
def augmentation(train_x,train_y):
    final_train_data = []
    final_target_train = []
    for i in tqdm(range(train_x.shape[0])):
        final_train_data.append(train_x[i])
        final_train_data.append(rotate(train_x[i], angle=45, mode = 'wrap'))
        final_train_data.append(np.fliplr(train_x[i]))
        final_train_data.append(np.flipud(train_x[i]))
        final_train_data.append(random_noise(train_x[i],var=0.1**2))
        for j in range(5):
            final_target_train.append(train_y[i])
    return final_train_data, final_target_train

# converting training images into torch format
def convert(final_train_data,final_target_train,val_x,val_y):

    final_train = np.array(final_train_data)
    final_target_train = np.array(final_target_train)

    final_train = final_train.reshape(2415, 3, 224, 224)
    final_train  = torch.from_numpy(final_train)
    final_train = final_train.float()

    # converting the target into torch format
    final_target_train = final_target_train.astype(float)
    final_target_train = torch.from_numpy(final_target_train)
    
    # converting validation images into torch format
    val_x = val_x.reshape(121, 3,224, 224)
    val_x  = torch.from_numpy(val_x)
    val_x = val_x.float()

    # converting the target into torch format
    val_y = val_y.astype(float)
    val_y = torch.from_numpy(val_y)


    return final_train, final_target_train, val_x, val_y
