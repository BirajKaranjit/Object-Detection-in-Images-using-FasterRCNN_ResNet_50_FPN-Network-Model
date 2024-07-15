# Object-Detection-in-Images-using-FasterRCNN_ResNet_50_FPN-Network-Model

**Introduction**
In the field of autonomous driving, Object Detection in Images and in Real-Time scenario is a crucial task that enables self-driving cars to perceive and interact with their environment 
safely. This project aims to develop a deep learning model for object detection using convolutional neural networks (CNNs) specifically saying FASTER_RCNN with ResNet_50_FPN model as the 
backbone. The dataset consists of images captured from a front-facing camera mounted on a car, featuring various objects such as pedestrians, cars, and traffic signs.

**Data Exploration**
The KITTI Dataset is used which consists of object detection and object orientation estimation benchmark consists of 7481 training images and 7518 test images, comprising a total of
80.256 labeled objects.

*Loading and Exploring the Dataset*
The dataset was loaded from a specified directory on Google Drive, consisting of images and corresponding annotations. The images are in RGB format, and the annotations provide bounding
box coordinates and class labels for the objects present in each image.
In order to feed the datasets to the Object Detection Model, the KITTI dataset’s labels has to be preprocessed in the specific format that is acceptable by the pre-trained model previously
on the COCO (Common Object in Context) datasets.
The Dataset’s sample is attached here with:

![Datasets_Sample1](https://github.com/user-attachments/assets/e2b118b5-4d91-4902-ad80-e40d65c0f682)
![Datasets_Sample2](https://github.com/user-attachments/assets/9ea36fcc-7332-46e3-87ff-d67d196ca94a)
![Datasets_Sample3](https://github.com/user-attachments/assets/4034bf60-2f55-4622-ad6c-e6de9abede4d)
I performed the feature extraction in the available datasets labels and used only the class_names and the bounding boxes coordinates to locate the object in the images by the model.

*Datasets Class Distribution*
An initial exploration of the dataset revealed the distribution of object classes. The dataset contains the following classes: Pedestrian, Car, Van, Truck, Traffic Sign, Cyclist etc.
comprising of the 8 different classes and +1 for background class.
Previously, the COCO dataset has 91 classes and later on the model is finetuned to have 8 different classes as per the requirement mentioned by the qualification task.
