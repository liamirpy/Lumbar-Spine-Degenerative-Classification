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


For executing the code run : 

``` 

python3 cleaning_csv.py 

```

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
