# Lumbar-Spine-Degenerative-Classification

This repository focuses on developing models for the detection and classification of degenerative spine conditions using lumbar spine MR images.

he aim is to simulate a radiologist's diagnostic process, helping to identify and assess the severity of conditions such as neural foraminal narrowing, subarticular stenosis, and spinal canal stenosis. This project is part of a competition to build models that enhance the accuracy and efficiency of diagnosing these spine conditions


This repository is for solving the kaggle competition and the link of this competition is : 

[RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/overview)


You can download the dataset from the link above.




# Data Description 

As explained above the aim of this challenge is that to classify the lumbar degenerative spine codition.
Based on the overview of this challenge we have 5 lumbar spine degenerative:

- Right Neural Foraminal Narrowing
- Left Neural Foraminal Narrowing
- Left Subarticular Stenosis
- Right Subarticular Stenosis
- Spinal Canal Stenosis


This codition can ocure across 5 intervertebral disc level:

- L1/L2
- L2/L3 
- L3/L4 
- L4/L5
- L5/S1

for this condition we want to predict severity scores and the score are : 

- Normal/Mild
- Moderate
- Severe


This dataset collected from different sources and we different modalities and view, there are 3 view and modalities:

- Sagittal T1
- Sagittal T2
- Axial T2 

Based on this dataset for each view and modalities we have a specifict condition:

- Neural Foraminal Narrowing: Sagittal T1

- Spinal Canal Stenosis: Sagittal T2

- Subarticular Stenosis: Axial T2 



In this dataset, we have a csv file that explain the severity scores, the modalities of data, and the train labels coordination (x,y).
All this description are in 3 different csv files.




# Approach for Training and Evaluation

For training and evaluation of this dataset, there are several possible approaches. I want to explain the approach I plan to develop.

First, I aim to develop a detection model. When classifying the entire image, much of the data is irrelevant and doesn't contribute to model training. Therefore, the first step is to develop a detection model that identifies and draws a bounding box around the specific part of the image we want to classify.

To achieve this, I will use the YOLO v8 pre-trained model as the detection model. (I will explain more details in the following sections of the README.)

In the second step, I will develop a new classifier model to train exclusively on the specific part of the image defined by the bounding box.

Overall Process of the Model
The model will follow a sequential approach:

Detection: Identify the region of interest (ROI) in the first step and extract that part from the entire image.
Classification: Pass the extracted ROI through the classifier model to predict the severity score



## Condition Detector Model


One of the most popular models for segmentation and detection is YOLO, which is trained on large datasets and is known for its speed.

For each condition, we develop this model separately, and for each condition, we have a different number of output classes.

[!IMPORTANT]
We train the model for each condition separately, effectively separating the view and condition, because for each condition, we use the same view and modality.

The models we are going to train, along with the number of output classes for each, are as follows:



|           Condition            |           Number of Output       |    
| -------------------------------| -------------------------------- |
|   Neural Foraminal Narrowing   | 2(left/right) * 5(Disc level)= 10| 
|     Spinal Canal Stenosis      |       1 * 5(Disc level)= 5       |
|     Subarticular Stenosis      | 2(left/right) * 5(Disc level)= 10|





> [!IMPORTANT]
> We can also add the score severity in number of class, but we want to the model consentrate in this classis and use other model for classifing the severity.



## Score Classifier Model

For the classifier model, we plan to use the pre-trained VGG-19 model for classification.

Similar to the detector model, we will develop three separate models, one for each condition. (Note that left and right sides are treated as part of the same condition and model.)







# Data Cleaninig

First of all, we are working with the train_label_coordinates.csv file which is includes :

- study_id
- series_id
- instance_number
- condition
- level
- x,y

Because we only want to train and work with the part of image that we have a coordintation (The box).

In this part we add also the other information that available for this dataset and to this csv file and 
create and save new csv file. The data that we want to add to this csv file are:

- series_description (from: train_series_descriptions.csv file)

- severity score (from: train.csv file)


The code for this purpose are available in directory [Data_Cleaning](./Data_cleaning/)directory.



> [!IMPORTANT]
> You need to put the data (all the folders and csv file) in the main directory of this repository.


The output is the csv file named:

 dataset_description.csv


 the columns are : 

 - study_id
 - series_id
 - instance_number
 - condition
 - level
 - x
 - y
 - series_description
 - score


# Data condition Spliting

After the previous part that we create a single csv file that includes all the data that we want to use it, 
now we want to splitted this data to seperated csv file for each condition. (The left and right condition are in same group)


At the final we have three csv file:


- Neural_Foraminal_Narrowing.csv
- Subarticular_Stenosis.csv 
- Spinal_Canal_Stenosis.csv


The code for this purpose are available in directory [Data_Condition_Splitting](./Data_splitting/)directory.




















# Condition Detector Model (cross-validation,preparing data, training)


For training and evaluation model, the various cross-validation approach developed. For this dataset, we 
develop k-fold cross-validation. The fold seperated based on the classes (condition + level)

After we splitted the data based on the k-fold, we want to prepare data for training.
For preparing the data for YOLO v.8 the data should be in this structure:


datasets/
│
├── train/
│   ├── images/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
│
├── val/
│   ├── images/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
│
└── yolo_config.yaml

we convert the dcm to png with normalization, and we calculate the label for each image and save it as .txt format.

The format of label should be :

class_id center_x center_y width height



> [!IMPORTANT]
> The model that we want to use is YOLO version 8.0. 


The code for this purpose are available in directory [Detector_Model](./Detector_data/)directory.




