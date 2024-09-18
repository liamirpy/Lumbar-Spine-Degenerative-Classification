'''


In this code we splitted the data based on the classed distiribution to 5 folds



'''


import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Read the CSV file


def splitted_data(condition):
        
    df = pd.read_csv(f'./condition_csv/{condition}.csv')

    # Check if 'score' column exists and map the score values to categorical values if needed
    # Assuming the scores are already labeled as 'Normal', 'Moderate', and 'Severe'
    # If they are numeric (1, 2, 3), convert them to labels
    # df['score'] = df['score'].map({1: 'Normal', 2: 'Moderate', 3: 'Severe'})

    # Initialize the StratifiedKFold with 5 splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Add a new column for fold assignment
    df['fold'] = -1  # Initialize with -1

    # Perform the stratified split
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['score'])):
        df.loc[val_idx, 'fold'] = fold  # Assign the fold number to the validation indices

    # Save the result to a new CSV file
    df.to_csv(f'./{condition}_folds.csv', index=False)

    print("CSV file with folds has been saved.")




splitted_data('Neural_Foraminal_Narrowing')