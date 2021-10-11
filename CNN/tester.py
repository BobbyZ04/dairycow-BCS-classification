# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:09:36 2021

@author: BOZ
"""
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import torch

def Tester(test_img_path, data, model):
    imgs = data['image_names']
    cropped_imgs = []
    for img_name in tqdm(imgs):
        img_path =  test_img_path + img_name
        img = imread(img_path)
        img_resized = resize(img,(224,224),anti_aliasing=True)
        cropped_imgs.append(img_resized)

    whole_x = np.array(cropped_imgs)
    whole_y = data['new_labels'].values
    
    whole_x = whole_x.reshape(len(whole_y), 3, 224, 224)
    
    #transform the array data to tensors
    whole_x  = torch.from_numpy(whole_x)
    whole_x = whole_x.float()
    #use models to make predictions
    predicted = model(whole_x)
    
    softmax_test = torch.exp(predicted)
    prob_test = list(softmax_test.detach().numpy())
    predictions_test = np.argmax(prob_test, axis=1)
    
    return whole_y, predictions_test