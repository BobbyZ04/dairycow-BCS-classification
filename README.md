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
Assume error range within 0.25 as the precise estimation, error range within 0.5 is the rough estimation. The precise estimation result is about <mark> 60% </mark>, and rough precise estimation is about <mark> 85% </mark>.

![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/images/rollingball_result1.png)
![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/images/rollingball_result2.png)

![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/images/rollingball_result3.png)

## CNN models
Applied several specific image processing techniques to the single-channel depth images, multplied them to rgb like 3-channel images. Made a whole new dataset:

_**Normalization to 0-255**:_ to cancel the influence of the real-world scale.

_**Fourier transformation with a high-pass filter**:_ to keep the sharp high frequency information like the bone frames.

_**Canny edge detection**:_ to detect the contours and edges on the back and tails.

![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/multi-channels.png)

After deviding the dataset to training - testing with 80% - 20%, applied augmentation to the training set, enlarged it to around 2300 images.

Tried Squeezenet, CNN1, CNN2, turned out the simpliest model has the best results, avoiding overfitting well considering such a limited dataset.
### results
Assume error range within 0.25 as the precise estimation, error range within 0.5 is the rough estimation. The precise estimation result is about <mark> 75% </mark>, and rough precise estimation is about <mark> 100% </mark>.

![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/images/CNN_acc.png)
![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/images/CNN_loss.png)

![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/images/training_cm.png)
![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/images/testing_cm.png)

### RGB - depth coordinates mapping
Kinect v2 uses different cameras for RGB and depth images, there are distortions between these two different views, also different size images. To make good collaborative analysis of RGB and depth information. Coordinates matching is needed.
If using Kinect SDK [repo](https://github.com/microsoft/Azure-Kinect-Sensor-SDK), one command should achiecve the goal:
```
MapDepthFrameToColorSpace(512 * 512, depthData, 512 * 512, m_pColorCoordinates);
```
where depthData is the depth image,  m_pColorCoordinates is the result mapped from depth to color space.
I didn't have the camera device with me so were not able to use SDK or do the precise calibration manually. I took example from [this blog](https://www.lhyd.top/archives/339182.html), applied their parameters directly to our images, so the matching result below looks pretty sketchy. 

![](https://github.com/BobbyZ04/dairycow-BCS-classification/blob/main/images/coordinate_matching.JPG)

If I got the device all set up I would try both SDK and mannual calibration, should be able to aquire good results.

## Next Steps
Enlarge dataset, compare more sophisticated DL models, try YOLO/RCNN for detecting in the wild eventually to achieve fully automatic BCS estimation.
