

import pandas as pd
from sklearn.model_selection import StratifiedKFold




def cross_validation(csv_file,results_name):


# Load your CSV file into a pandas DataFrame
    df = pd.read_csv(f'{csv_file}')

    # Concatenate 'condition' and 'level' to create a unique class for each combination
    df['condition_level'] = df['condition'] + '_' + df['level']

    # Now, we can assign numeric class labels if needed
    df['class_id'] = df['condition_level'].astype('category').cat.codes

    # Prepare for 5-fold stratified split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Split the data
    df['fold'] = -1  # Initialize the fold column

    # Assign fold numbers
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['class_id'])):
        df.loc[val_idx, 'fold'] = fold

    # Now, 'df' contains a 'fold' column that indicates the fold assignment (0-4)
    # Save the new CSV with fold numbers if necessary
    df.to_csv(f'{results_name}.csv', index=False)

    print("Data has been split into 5 folds and saved as 'your_file_with_folds.csv'")



cross_validation('../../Data_Condition_Splitting/Spinal_Canal_Stenosis.csv','Spinal_Canal_Stenosis_folds')

cross_validation('../../Data_Condition_Splitting/Neural_Foraminal_Narrowing.csv','Neural_Foraminal_Narrowing_folds')

cross_validation('../../Data_Condition_Splitting/Subarticular_Stenosis.csv','Subarticular_Stenosis_folds')
