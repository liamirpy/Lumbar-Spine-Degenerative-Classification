

'''
In this code we want to splitting the csv file data to three sperated csv file based on the condition.

'''





csv_file='../Data_Cleaning'

import pandas as pd

# Load the original CSV file
df = pd.read_csv(f'{csv_file}/dataset_description.csv')

# Define conditions for each group
condition_groups = {
    'Spinal Canal Stenosis': ['Spinal Canal Stenosis'],
    'Neural Foraminal Narrowing': ['Right Neural Foraminal Narrowing', 'Left Neural Foraminal Narrowing'],
    'Subarticular Stenosis': ['Right Subarticular Stenosis', 'Left Subarticular Stenosis']
}

# Split and save to separate CSV files
for group_name, conditions in condition_groups.items():
    # Filter rows based on condition
    filtered_df = df[df['condition'].isin(conditions)]
    # Save to new CSV file
    group_name_save=group_name.replace(' ','_')
    filtered_df.to_csv(f'{group_name_save}.csv', index=False)

print("CSV files have been split and saved.")




