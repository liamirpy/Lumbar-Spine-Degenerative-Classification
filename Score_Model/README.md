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


The output is name of condition folders.


The code for this purpose are available in directory [data_prepration](./Detector_data/)directory.


4. Create a label csv file : For this purpose we generate the the labels for each condition and folds.


The output are in the condtion_labels folders.

The code for this purpose are available in directory [data_prepration](./Detector_data/)directory.




4. Model Training: Finally, we trained the model using the prepared data.




## K-Fold Cross-Validation Results for Neural Foraminal Narrowing

During the training of the model, we used **K-Fold Cross-Validation** to evaluate performance across multiple subsets of the data. Below are the evaluation metrics for each fold:

| Fold  | Accuracy  | Precision | Recall    | F1 Score  |
|-------|-----------|-----------|-----------|-----------|
| Fold 0| 0.7801    | 0.8346    | 0.7801    | 0.7984    |
| Fold 1| 0.7852    | 0.8385    | 0.7852    | 0.8018    |
| Fold 2| 0.8121    | 0.8208    | 0.8121    | 0.8161    |
| Fold 3| 0.7976    | 0.8292    | 0.7976    | 0.8096    |
| Fold 4| 0.7978    | 0.8310    | 0.7978    | 0.8105    |

### Best Performing Fold

Based on the results, **Fold 2** is selected as the best-performing model. Here's why:

- **Highest Accuracy:** Fold 2 achieved the highest accuracy of **0.8121**, indicating it correctly classified the most samples overall.
- **Best F1 Score:** Fold 2 also has the highest F1 Score of **0.8161**, which is a balance between precision and recall, making it a strong candidate for tasks where both false positives and false negatives are of concern.
- **Balanced Performance:** While Fold 1 had a slightly higher precision, Fold 2’s overall balance across accuracy, precision, recall, and F1 score makes it the most reliable fold for the final model selection.

This model will be used for further evaluation and deployment due to its superior performance.





## K-Fold Cross-Validation Results for Spinal Canal Stenosis

During the training of this model, we used **K-Fold Cross-Validation** to evaluate performance across multiple subsets of the data. Below are the evaluation metrics for each fold:

| Fold  | Accuracy  | Precision | Recall    | F1 Score  |
|-------|-----------|-----------|-----------|-----------|
| Fold 0| 0.8954    | 0.9065    | 0.8954    | 0.9002    |
| Fold 1| 0.8288    | 0.9083    | 0.8288    | 0.8586    |
| Fold 2| 0.8683    | 0.9062    | 0.8683    | 0.8837    |
| Fold 3| 0.8928    | 0.9098    | 0.8928    | 0.8997    |
| Fold 4| 0.8995    | 0.9120    | 0.8995    | 0.9051    |

### Best Performing Fold

Based on the results, **Fold 4** is selected as the best-performing model. Here's why:

- **Highest Accuracy:** Fold 4 achieved the highest accuracy of **0.8995**, indicating it correctly classified the most samples.
- **Best F1 Score:** Fold 4 also has the highest F1 Score of **0.9051**, which shows its balance between precision and recall.
- **Highest Precision:** Fold 4 also achieved the highest precision of **0.9120**, meaning it minimizes false positives.

This fold demonstrates superior performance across all key metrics, making it the best candidate for further evaluation and potential deployment.




## K-Fold Cross-Validation Results for Subarticular Stenosis

We used **K-Fold Cross-Validation** to evaluate the performance of this model across multiple data subsets. Below are the evaluation metrics for each fold:

| Fold  | Accuracy  | Precision | Recall    | F1 Score  |
|-------|-----------|-----------|-----------|-----------|
| Fold 0| 0.7903    | 0.8212    | 0.7903    | 0.8017    |
| Fold 1| 0.7494    | 0.8132    | 0.7494    | 0.7700    |
| Fold 2| 0.7931    | 0.8199    | 0.7931    | 0.8034    |
| Fold 3| 0.7164    | 0.8205    | 0.7164    | 0.7447    |
| Fold 4| 0.7205    | 0.8085    | 0.7205    | 0.7468    |

### Best Performing Fold

Based on the results, **Fold 2** is selected as the best-performing fold. Here's why:

- **Highest Accuracy:** Fold 2 achieved the highest accuracy of **0.7931**, indicating that it correctly classified the most samples overall.
- **Best F1 Score:** Fold 2 also had the best F1 Score of **0.8034**, providing a balance between precision and recall.
- **Balanced Precision and Recall:** Fold 2 has a strong balance between **precision (0.8199)** and **recall (0.7931)**, making it reliable for both minimizing false positives and maximizing correct classifications.

This fold will be used for further evaluation and potential deployment due to its overall balanced and superior performance.





