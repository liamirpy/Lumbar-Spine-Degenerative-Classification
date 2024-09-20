# Score Model (Data augmentation, data prepration , training)

In the previous section, we developed the YOLO v8 model for classifying conditions. In this section, we focus on developing another classifier for predicting the severity score (Normal, Moderate, Severe).

For this purpose, we implemented a VGG-based model.

Additionally, due to the imbalance in the number of samples across the classes, we explored two approaches:

1. Data augmentation
2. A custom loss function based on class distribution

## Data Augmentation

Data augmentation is one approach to handling class imbalance. However, the challenge here is that excessive data augmentation can lead to overfitting. In this dataset, the class imbalance is severe for each condition. If we apply too much data augmentation, the model may overfit.

To address this, we approached the imbalance as follows: We have three classes—one majority class and two minority classes. Our strategy is to augment the first minority class until it reaches one-third of the size of the majority class, and augment the second minority class until it reaches half the size of the majority class.

We then applied a custom loss function designed to account for the class distribution.




## Cross_validation

For cross-validation, we split the data into k-folds. In each fold, the split was done based on the dataset distribution, meaning that in the first step, we divided the data into k-folds where each fold maintained the same class distribution as the total dataset for that condition. Data augmentation was then applied to the training folds, but not to the evaluation fold. This ensures that the evaluation data remains untouched by augmentation.



The output :

- condition
    - fold_0
        - condition_train.csv
        - condition_val.csv
    
    .
    .
    .
    - fold_4
        - condition_train.csv
        - condition_val.csv



## Apprach 

1. Data Splitting: We first split the data into 5 folds, ensuring that each fold maintains the same class distribution as the overall dataset for each condition. This process is part of the cross-validation folder, and the results are stored in CSV files.

The code for this purpose are available in directory [cross_validation](./Detector_data/)directory.




2. Data Augmentation: Data augmentation was applied to the training data. For example, in the first fold, the first fold is used as the validation set, and augmentation is applied to the other folds. The augmented data is recorded in the CSV file, with an additional column indicating the type of augmentation used for each sample.


The code for this purpose are available in directory [data_augmentation](./Detector_data/)directory.



> [!IMPORTANT]
> We have three classes—one majority class and two minority classes. Our strategy is to augment the first minority class until it reaches one-third of the size of the majority class, and augment the second minority class until it reaches half the size of the majority class.




3. DCM to PNG: in this part, we read each csv file and convert all the row to png format and if there is any data augmentation approach we also apply that augmentation approach for that subjects. 

The code for this purpose are available in directory [data_prepration](./Detector_data/)directory.



4. Model Training: Finally, we trained the model using the prepared data.


