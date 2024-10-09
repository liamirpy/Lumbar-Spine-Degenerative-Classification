'''
In this code we convert dcm to png for each condition and its folds


'''





import pandas as pd


from pathlib import Path



def working_on_csv(csv_directory,train_or_val):


    df = pd.read_csv(csv_directory)



    # Function to create the 'subject' column
    def create_subject(row):
        base_name = f"{row['study_id']}_{row['series_id']}_{row['instance_number']}_{int(row['x'])}_{int(row['y'])}.png"

        if train_or_val=='train':

            if pd.notna(row['augmentation']) and row['augmentation'] != '':
                return base_name.replace('.png', '_augmented.png')

        return base_name

    # Function to create the 'label' column
    def create_label(score):
        if score == 'Normal/Mild':
            return 1
        elif score == 'Moderate':
            return 2
        elif score == 'Severe':
            return 3
        return None

    # Apply the functions to the DataFrame
    df['subject'] = df.apply(create_subject, axis=1)
    df['label'] = df['score'].apply(create_label)

    # Select the relevant columns
    new_df = df[['subject', 'label']]

    return new_df

    # Write the new DataFrame to a CSV file
   




    
def create_folder(condition):



    folder_path = Path(f'./{condition}_label')

    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)




    for fold in range(5):

        fold_path = Path(f'./{condition}_label/fold_{fold}')

        if not fold_path.exists():
            fold_path.mkdir(parents=True, exist_ok=True)







csv_directory= '../data_augmentation/augmented_output'

def create_label(condition):

    for fold in range(5):


        train_csv_path=f'{csv_directory}/{condition}/fold_{fold}/{condition}_augmented.csv'
        val_csv_path=f'{csv_directory}/{condition}/fold_{fold}/{condition}_val.csv'


        print(train_csv_path)


        train_label_csv=working_on_csv(train_csv_path,'train')

        val_label_csv=working_on_csv(val_csv_path,'val')



        ### saving 


        train_output_path=f'./{condition}_label/fold_{fold}/{condition}_augmented_labels.csv'

        train_label_csv.to_csv(train_output_path, index=False)


        val_output_path=f'./{condition}_label/fold_{fold}/{condition}_val_labels.csv'

        val_label_csv.to_csv(val_output_path, index=False)




        print(f'{condition}/{fold} done')









for con in ['Spinal_Canal_Stenosis','Neural_Foraminal_Narrowing','Subarticular_Stenosis']:


    create_folder(con)
    create_label(con)







