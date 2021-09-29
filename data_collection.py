# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:36:48 2021

@author: BOZ
"""
import pandas as pd
import os

import glob
import cv2
import numpy as np
#from matplotlib import pyplot as plt
#cropping images from videos

def read_img(image_file):
    imgOri = cv2.imread(image_file,-1)
    cow = cv2.split(imgOri)[0]
    return cow



def rescale(img):
    vmin = img.min()
    vmax = img.max()
    img_x = np.where(img==0,255,255*(img-vmin)/(vmax-vmin))
    return img_x.astype(dtype = np.uint8)


def cow_back_extraction(background,img_dir1, img_dir2,row_up, row_down, col_left, col_right,
                        pixcel_amount, tail, neck, height_high, height_low):
    bg_file = os.path.join(img_dir1,background)
    background = read_img(bg_file)
    background = background[row_up:row_down,col_left:col_right]
    
    #kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    #kernel_2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    
    for i, cow_img in enumerate(glob.glob(os.path.join(img_dir2,'*.png'))):
        #cow_org = cv2.imread(cow_img,-1)
        #cow_org = cv2.split(cow_org)[0]
        cow_org = read_img(cow_img)
        cow_crop = cow_org[row_up:row_down,col_left:col_right]
        subs_crop = pd.DataFrame(background - cow_crop)
        subs_crop[subs_crop>height_high] = 0
        subs_crop[subs_crop<height_low] = 0
        filtered = np.array(subs_crop)
        #print(np.shape(filtered))
        for j in range(0,int((filtered.shape[0])*0.4)):
            if np.sum(np.count_nonzero(filtered,axis=1)[j:j+3]) == 0 and np.count_nonzero(filtered)>=pixcel_amount: #and np.count_nonzero(filtered, axis=0)[0] == 0:        
                for k in range(row_down-row_up):
                    if np.count_nonzero(filtered,axis=1)[k+j]>0:
                        filtered = filtered [k+j-tail:k+j+neck,:] #shape should be 116*106
                        break
        
                cropped_filtered0 = cv2.medianBlur(filtered,5)
                #normalize the value to 0-255
                cropped_filtered = rescale(cropped_filtered0)
                
                #dilated = cv2.dilate(cropped_filtered,kernel_1)
                #ero = cv2.erode(dilated,kernel_2)
                #cropped_filtered = ero
                #print(np.shape(Cropped_filtered))
                
                #cv2.imwrite('second_run/cropped_depth2/'+ (os.path.basename(cow_img)),cropped_filtered)
                #cv2.imwrite('first_run/cropped_depth1/'+ (os.path.basename(cow_img)),cropped_filtered)
                #cv2.imwrite('second_run/hf_cropped/'+ (os.path.basename(cow_img)),cropped_filtered)

                lap = cv2.Laplacian(cropped_filtered,cv2.CV_64F)
                lap_result = cv2.convertScaleAbs(lap)
                #cv2.imwrite('second_run/lap2/'+ (os.path.basename(cow_img)),lap_result)            
                #cv2.imwrite('second_run/hf_lap/'+ (os.path.basename(cow_img)),lap_result)

                f = cv2.dft(cropped_filtered.astype(np.float32))
                f_shf = np.fft.fftshift(f)
                crow, ccol = np.int8((neck+tail)/2), np.int8((col_right-col_left)/2)
                mask = np.ones(((neck+tail), (col_right-col_left)),np.uint8)
                mask[crow-3:crow+3, ccol-2:ccol+8] = 0
                f_shf = f_shf*mask
                f_inverse = np.fft.ifftshift(np.float32(f_shf))
                f_back0 = cv2.idft(np.float32(f_inverse))
                f_back = cv2.medianBlur(f_back0, 5)
                f_back = rescale(f_back)
                #cv2.imwrite('second_run/Fourier2/'+ (os.path.basename(cow_img)),f_back)
                #cv2.imwrite('second_run/hf_Fourier/'+ (os.path.basename(cow_img)),f_back)
                
                channels = np.array([cropped_filtered,f_back,lap_result])
                #channels shape is (3,103,106), use Transpose to turn it into (103,106,3)
                channels = channels.transpose(1,2,0)
                cv2.imwrite('CNN_dataset/first_run/'+ (os.path.basename(cow_img)),channels)

def cow_delete(cow_dir,ID_list,max_number):
    '''delete the extra images of cows who have more than 5 images'''
    filelist = os.listdir(cow_dir)
    #keep only the last 5 images
    filelist.reverse()
    number_list = [0]*len(ID_list)
    
    for file in filelist:
        oldpath = os.path.join(cow_dir,file)
        filename = os.path.splitext(file)[0]
        for i,ID in enumerate(ID_list):
            if str(ID) in filename:
                number_list[i] += 1
                if number_list[i]>max_number:
                    os.remove(oldpath)


#label the images
def labelize(img_dir, form, save_path):
    labels = []
    #IDs = []
    image_names = []
    path_list=os.listdir(img_dir)
    #im_path = os.path.join(img_dir, '*.png')
    #for i, cow_img in enumerate(glob.glob(im_path)):
    for cow_img in path_list:
        for ID in form.Cow_ID:
            if str(ID) in cow_img:
                #here adapt the code to fill the labels you defined yourself
                labels.append(round(float(form.BCS_average[form.Cow_ID == ID])*4)/4)
                image_names.append(cow_img)
    Results = pd.DataFrame({'image_names':image_names, 'labels':labels})
    Results['new_labels'] = ''
    Results['new_labels'][Results.labels == 2.5] = 0
    Results['new_labels'][Results.labels == 2.75] = 1
    Results['new_labels'][Results.labels == 3.0] = 2
    Results['new_labels'][Results.labels == 3.25] = 3
    Results['new_labels'][Results.labels == 3.5] = 4
    Results['new_labels'][Results.labels == 4.0] = 5
    
    Results.to_csv(save_path,index=False)
    
    return Results




#BCS_data = pd.read_excel('BCS & Weights from 21Jul21.xlsx',sheet_name = 'BCS')
#img_dir = 'train_test_sep/new-train/'


