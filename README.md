# Object-Detection-in-Images-using-FasterRCNN_ResNet_50_FPN-Network-Model

**Introduction**
In the field of autonomous driving, Object Detection in Images and in Real-Time scenario is a crucial task that enables self-driving cars to perceive and interact with their environment 
safely. This project aims to develop a deep learning model for object detection using convolutional neural networks (CNNs) specifically saying FASTER_RCNN with ResNet_50_FPN model as the 
backbone. The dataset consists of images captured from a front-facing camera mounted on a car, featuring various objects such as pedestrians, cars, and traffic signs.

**Data Exploration**
The KITTI Dataset is used which consists of object detection and object orientation estimation benchmark consists of 7481 training images and 7518 test images, comprising a total of
80.256 labeled objects.

The google drive link to the datasets and the 'ModelFile'.pth: [Resources Link](https://bit.ly/3S7Wog6 )

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

*Preprocessing Steps*
To prepare the dataset for training, initially, the Image resizing and Normalization preprocessing techniques were applied but later on the output of the model seems to be better without the resizing and normalization functions. Thus, no any preprocessing techniques needed to be applied as the datasets are already in the appropriate format.

**Model Development**
Choosing a Pre-trained CNN Architecture
For this project, the Faster R-CNN model with a ResNet-50 backbone was selected and PyTorch Framework of ML is implemented to build the model. Faster R-CNN is a popular object detection model that combines a region proposal network (RPN) with a Fast R-CNN detector.
There are also other state-of-the-art models like YOLO_Vx available but I choose to implement this model for making the training process computationally efficient.
The model architecture is presented below:
![Faster-RCNN_ResNet_50_Architecture](https://github.com/user-attachments/assets/cc22413e-8c29-4dd4-b58f-a5d58d0d7bca)

*Fine-Tuning the Model*
The pre-trained ResNet-50 model was fine-tuned on the given dataset. The following modifications were made:

*Output Layer Adjustment*: The final fully connected layer i.e. the output layer is adjusted to predict bounding boxes and class probabilities for the specific object classes in the dataset and adjusted to predicted the objects according to the KITTI datasets.
The GOOGLE-COLAB is used as the text-editor as it provides free GPU computational resources.


**Evaluation**
Performance Evaluation
The model's performance was evaluated on the testing set using the following metrics:
The results are as follows:
![Model's Performance](https://github.com/user-attachments/assets/344a30f1-318a-4008-8451-0f63978cd0d8)
This is the performance metrices values of the model, furthermore the accuracy along with other metrices is improved.

**Visualization**
*Model Predictions*
The model's predictions were visualized on a few sample of local images from the testing set. The visualizations include bounding boxes and class labels with confidence scores. Below are some examples:

![ringroad](https://github.com/user-attachments/assets/e18c7f1c-ee94-4f12-bc53-9b726a6b231b)
![ringroad_result_img](https://github.com/user-attachments/assets/cff53987-08c2-417d-8ed0-0891db301f4e)
![ktm](https://github.com/user-attachments/assets/c46521e7-42cb-435a-8142-eb0c5d5fa5cc)
![ktm_result_img](https://github.com/user-attachments/assets/1cfb93fc-e1a1-45ab-b5b9-8eace7bdc1d9)

![LocalSample](https://github.com/user-attachments/assets/27e45093-a3d6-492e-b63a-956d5fc98a71)
![result_image_2](https://github.com/user-attachments/assets/15ad8b0b-d255-495c-b199-8efd28bc9fbb)

**Optimization**
*Strategies for Improvement*
To further optimize the model's performance, the following strategies are used and suggested:

*Hyperparameter Tuning:* Experiment with different learning rates, batch sizes, and number of epochs to find the optimal settings.
*Model Architecture Adjustments:* Consider using more advanced architectures like Efficient-Det or YOLOv5 for potentially better performance.
This project successfully developed an object detection model for self-driving cars using a pre-trained Faster R-CNN with a ResNet-50 backbone. The model was fine-tuned on a dataset of front-facing car images, achieving a good F1 score on the testing set. Overall, this model is performing really well and predicting the class name labels and bounding boxes accurately and within the appropriate image object boundary. 

*Potential Areas for Improvement*
•	Data Augmentation: Increasing the variety and complexity of data augmentation techniques.
•	Hyperparameter Optimization: Conducting a more thorough hyperparameter search.
•	Advanced Architectures: Exploring more advanced detection models like Efficient-Det and YOLOv5.


