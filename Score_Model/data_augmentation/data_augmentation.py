import pandas as pd
import numpy as np
import random
import shutil

import os





augmentations = ['Rotation', 'Flip', 'Noise']



def augment_data(df, augmentations):
    # Count the number of each class (normal, moderate, severe)
    class_counts = df['score'].value_counts()
    majority_class = class_counts.idxmax()  # Find the majority class
    minority_classes = class_counts[class_counts < class_counts[majority_class]].index  # Find minority classes

    # Set augmentation targets
    target_1 = class_counts[majority_class] // 3  # First minority class target: 1/3 of majority class
    target_2 = class_counts[majority_class] // 2  # Second minority class target: 1/2 of majority class

    def create_augmented_rows(minority_class, target):
        unique_minority = df[df['score'] == minority_class].drop_duplicates()
        unique_count = len(unique_minority)
        num_needed = max(0, target - class_counts[minority_class])  # Calculate how many new rows we need to add

        if unique_count >= num_needed:
            # Use unique samples first
            augmented_rows = unique_minority.sample(num_needed, replace=False)
        else:
            # Use all unique samples and repeat if necessary
            augmented_rows = unique_minority.copy()
            num_repeats_needed = num_needed - unique_count
            repeated_samples = unique_minority.sample(num_repeats_needed, replace=True)
            augmented_rows = pd.concat([augmented_rows, repeated_samples], ignore_index=True)

        # Add an augmentation column and randomly assign augmentation techniques
        augmented_rows['augmentation'] = [random.choice(augmentations) for _ in range(len(augmented_rows))]

        return augmented_rows

    # Augment first minority class
    minority_1 = minority_classes[0]
    augmented_rows_1 = create_augmented_rows(minority_1, target_1)

    # Augment second minority class
    minority_2 = minority_classes[1]
    augmented_rows_2 = create_augmented_rows(minority_2, target_2)

    # Add a new column to the original dataframe (for subjects that don't get augmented)
    df['augmentation'] = None

    # Combine the original dataframe with the augmented rows
    augmented_df = pd.concat([df, augmented_rows_1, augmented_rows_2], ignore_index=True)

    return augmented_df






def augment_data(df, augmentations):


    # Step 1: Count the number of each class (normal, moderate, severe)
    class_counts = df['score'].value_counts()
    majority_class = class_counts.idxmax()  # Find the majority class

    minority_classes = class_counts[class_counts < class_counts[majority_class]].index  # Find minority classes

    # Step 2: Set augmentation targets
    target_1 = class_counts[majority_class] // 3  # First minority class target: 1/3 of majority class
    target_2 = class_counts[majority_class] // 2  # Second minority class target: 1/2 of majority class

    # List of augmentations

    # Step 3: Augment first minority class
    minority_1 = minority_classes[1]

    print('miniortiy ',minority_1)

    num_augments_1 = target_1 - class_counts[minority_1]  # Calculate how many new rows we need to add

    # Randomly choose subjects to augment from the first minority class
    augmented_rows_1 = df[df['score'] == minority_1].sample(num_augments_1, replace=True)

    # Add an augmentation column and randomly assign augmentation techniques
    augmented_rows_1['augmentation'] = [random.choice(augmentations) for _ in range(len(augmented_rows_1))]

    # Step 4: Augment second minority class
    minority_2 = minority_classes[0]
    print('miniortiy ',minority_2)

    num_augments_2 = target_2 - class_counts[minority_2]  # Calculate how many new rows we need to add

    # Randomly choose subjects to augment from the second minority class
    augmented_rows_2 = df[df['score'] == minority_2].sample(num_augments_2, replace=True)

    # Add an augmentation column and randomly assign augmentation techniques
    augmented_rows_2['augmentation'] = [random.choice(augmentations) for _ in range(len(augmented_rows_2))]

    # Step 5: Add a new column to the original dataframe (for subjects that don't get augmented)
    df['augmentation'] = None

    # Step 6: Combine the original dataframe with the augmented rows
    augmented_df = pd.concat([df, augmented_rows_1, augmented_rows_2], ignore_index=True)


    return augmented_df





main_folders = ['Neural_Foraminal_Narrowing', 'Spinal_Canal_Stenosis', 'Subarticular_Stenosis']

# Subfolders (fold_0 to fold_4)
sub_folders = [f'fold_{i}' for i in range(5)]

# Path to your root directory where "a", "b", "c" folders exist
root_dir = '../cross_validation'



for main_folder in main_folders:

    for sub_folder in sub_folders:
        # Construct the path to the train.csv in each subfolder
        csv_path = os.path.join(root_dir, main_folder, sub_folder, f'{main_folder}_train.csv')

        # Check if the file exists
        if os.path.exists(csv_path):
            # Read the CSV file
            df = pd.read_csv(csv_path)

            # Apply augmentation
            augmented_df = augment_data(df, augmentations)

            # Create the output folder if it doesn't exist
            output_folder = os.path.join('augmented_output', main_folder, sub_folder)
            os.makedirs(output_folder, exist_ok=True)

            # Save the augmented dataframe to a new CSV file in the corresponding folder
            output_csv_path = os.path.join(output_folder, f'{main_folder}_augmented.csv')
            augmented_df.to_csv(output_csv_path, index=False)

            print(f"Augmented data saved to: {output_csv_path}")


            ## also copy the val csv 


            val_csv = f'{root_dir}/{main_folder}/{sub_folder}/{main_folder}_val.csv'


            destination_dir = f'{output_folder}'


            destination_file = os.path.join(destination_dir, f'{main_folder}_val.csv')


            # Copy the file
            shutil.copy(val_csv, destination_file)




