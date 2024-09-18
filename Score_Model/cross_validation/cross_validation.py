'''


In this code we splitted the data based on the classed distiribution to 5 folds



'''
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


def splitted_data(condition):
        
    df = pd.read_csv(f'./condition_csv/{condition}.csv')


    # Find and delete rows where the 'score' column has NaN values
    df = df.dropna(subset=['score'])

    # Reset the index so it becomes continuous after dropping rows
    df = df.reset_index(drop=True)

    # Initialize the StratifiedKFold with 5 splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Add a new column for fold assignment
    df['fold'] = -1  # Initialize with -1

    # Perform the stratified split
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['score'])):
        df.loc[val_idx, 'fold'] = fold  # Assign the fold number to the validation indices

    # Save the result to a new CSV file


    condition_folder = Path(f'./{condition}')
    os.makedirs(condition_folder, exist_ok=True)

    df.to_csv(f'./{condition}/{condition}_folds.csv', index=False)



    for fold in range(5):
            

        fold_folder = os.path.join(f'./{condition}', f'fold_{fold}')
        os.makedirs(fold_folder, exist_ok=True)
        
        # Split data into train and validation sets
        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]
        
        # Save train and validation CSV files for the current fold
        train_csv_path = os.path.join(fold_folder, f'{condition}_train.csv')
        val_csv_path = os.path.join(fold_folder, f'{condition}_val.csv')
        
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)




splitted_data('Neural_Foraminal_Narrowing')
# 
splitted_data('Spinal_Canal_Stenosis')
# 
splitted_data('Subarticular_Stenosis')

