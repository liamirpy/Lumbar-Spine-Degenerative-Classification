# Condition Detector Model (cross-validation,preparing data, training)


For training and evaluation model, the various cross-validation approach developed. For this dataset, we 
develop k-fold cross-validation. The fold seperated based on the classes (condition + level)



## Cross-validation

for cross-validation, we read each condition csv file and based on the classed that is condition+level, we seperated the data to K-fold. the k here is 5.

For each condition the single csv file will create and added two column : calss_id and folds.





## Data prepration

In this part we create a dataset that suitable for training and evaluation of YOLO models.



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



For each condition there is a folder that includes five subfolder (5 folds) and each subfolder there is datasets structure like above.


Due to the right of dataset, the dataset not includes here. 







## Training 


For training, go to the training directory and run the training.

The result for each condition and fold will save in results directory.



