
'''
In this code we use the train_series_descriptions.csv and add other information from other csv files and save  as 
a single csv file as a dataset_description.csv file and we only work with this csv file to the end of this repository.


ATTENTON: All the csv file should be put in folder of data in main directory.

'''





## Add series_description

Data_path='../Data'


import pandas as pd

# Read the two CSV files
train_label_coordinates = pd.read_csv(f'{Data_path}/train_label_coordinates.csv')
train_series_descriptions = pd.read_csv(f'{Data_path}/train_series_descriptions.csv')

# Merge the two dataframes on 'study_id' and 'series_id'
merged_csv = pd.merge(train_label_coordinates, train_series_descriptions[['study_id', 'series_id', 'series_description']], 
                      on=['study_id', 'series_id'], how='left')

# Save the resulting dataframe to a new CSV file





## Add score 


import pandas as pd

# Step 1: Read the updated CSV file (file1_with_series_description.csv) and the third CSV file (Train.csv)
updated_df = merged_csv
train_df = pd.read_csv(f'{Data_path}/train.csv')

# Step 2: Define a function to fetch the score based on 'study_id' and 'condition'
def get_score(row):
    study_id = row['study_id']
    condition = row['condition']
    level = row['level']

    level_1=level.split('/')[0]
    level_2=level.split('/')[1]

    condition_level= f'{condition}_{level_1}_{level_2}'

    condition_level= condition_level.replace(' ','_')

    condition_level=condition_level.lower()
    
    
    # Check if the 'study_id' exists in the Train.csv and if the 'condition' is a valid column
    if study_id in train_df['study_id'].values and condition_level in train_df.columns:
        # Return the value in the 'condition' column for the given 'study_id'
        return train_df.loc[train_df['study_id'] == study_id, condition_level].values[0]
    else:
        return None  # Return None if no match is found

# Step 3: Apply the function to each row of the updated dataframe
updated_df['score'] = updated_df.apply(get_score, axis=1)

# Step 4: Save the updated dataframe with the new 'score' column to a new CSV file
updated_df.to_csv('dataset_description.csv', index=False)

print("New CSV with the 'score' column has been created.")






