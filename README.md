# dairycow-BCS-Estimation
This project is designed for dairy cow Body Condition Score automatically classification by Kinect v2 depth camera. Built an image dataset and implemented morphalogical and deep learning models to compare results.

![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/cow_BCS.jpg)

## Problem Describtion
Body Condition Score is a measure of relative amount of subcutaneous body fat or energy reserved in cows. It is widely used for evaluating dairy cow's health status and milk production situation. Nowadays most of the BCS measurements are done by very experienced professionals, which require high cost of human resouces, time and money. In this repo, we used Microsoft Kinect v2 3D-cameras to take videos of dairy cows and made our datasets and applied several different models on them.

## Data
I collected the rgb and depth image frames from videos where lots of noises like dirty floors, humans, fences are in the images. Then set some constraints on keeping useful images in data_collection.py. Then got the raw dataset with cow-back-only images, with a size around 800 images. For these two methods, only depth images were used for analysis.

## 3D-rolling ball model
The key part of this model is the idea that the BCS of a cow has positive correlation with the angularity of its back. More angularity means skinnier the cow is.
The main process is to calculate the angularity of some cows with standard BCS using an open operation, define a baseline and then apply a polynomial regression model. By inputting weight and angularity data to the model, we got the output BCS results. The process and results are in the 3D-rolling ball model.ipynb.
### results
![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/images/rollingball_result1.png)![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/images/rollingball_result2.png)
![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/images/rollingball_result3.png)

## CNN models
Applied several specific image processing techniques to the single-channel depth images, multplied them to rgb like 3-channel images. Made a whole new dataset:

_**Normalization to 0-255**:_ to cancel the influence of the real-world scale.

_**Fourier transformation with a high-pass filter**:_ to keep the sharp high frequency information like the bone frames.

_**Canny edge detection**:_ to detect the contours and edges on the back and tails.

![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/multi-channels.png)

After deviding the dataset to training - testing with 80% - 20%, applied augmentation to the training set, enlarged it to around 2300 images.
