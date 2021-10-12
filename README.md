# dairycow-BCS-Estimation
This project is designed for dairy cow Body Condition Score automatically classification by Kinect v2 depth camera. Built an image dataset and implemented morphalogical and deep learning models to compare results.

![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/cow_BCS.jpg)

## Problem Describtion
Body Condition Score is a measure of relative amount of subcutaneous body fat or energy reserved in cows. It is widely used for evaluating dairy cow's health status and milk production situation. Nowadays most of the BCS measurements are done by very experienced professionals, which require high cost of human resouces, time and money. In this repo, we used Microsoft Kinect v2 3D-cameras to take videos of dairy cows and made our datasets and applied several different models on them.

## Raw dataset
I collected the image frames from videos by ffmpeg, and then set some constraints on keeping valuable images in data_collection.py 

## 3D-rolling ball model
The key part of this model is the idea that the BCS of a cow has positive correlation with the angularity of its back. More angularity means skinnier the cow is.
The main process is to calculate the angularity of some cows with standard BCS, define a baseline and then apply a polynomial regression model. By inputting weight and angularity data to the model, we got the output BCS results.

