

import pandas as pd 

import numpy as np
from pathlib import Path

import pydicom
import os 

import cv2
import yaml

import csv

class Detector_data_prepration:
    

    def __init__(
        self,

        dataset_directory='../Data/train_images',

        csv_directory='',


        condition_level_classes={},

        condition_name='',

        val_fold=1,

        width_box=16,


        ):


        self.dataset_directory = dataset_directory

        self.csv_directory=csv_directory


        self.condition_level_classes=condition_level_classes

        self.condition_name = condition_name

        self.val_fold=val_fold

        self.width_box=width_box


        self.save_directory= f'./{self.condition_name}'







        ## Create a folder for saving data 
        self.create_folder()


        ## read data based on the cross validation and fold that define as a validation fold
        self.read_cross_validation()


        ### Convert train data to png for training data 
        self.dicom_to_png(self.training_data,self.train_image_path)
#
        self.save_weight_height_to_csv()

        self.create_label_for_yolo(self.training_data, self.train_labels_path)



        ### Convert train data to png for validation data 
        self.dicom_to_png(self.validation_data,self.val_images_path)
#
        self.save_weight_height_to_csv()

        self.create_label_for_yolo(self.validation_data, self.val_labels_path)




        self.creata_yaml_file()




        '''

            Due to the fact that in each study id we have a number of series id, and for each series id there are 
            number of instance id, we create a dictionary to read and load data in once to save time for loading the dcm.

            The data will create like this :


            for example :

                study_id,series_id,instance_number
                    1,       101,        1
                    1,       101,        2
                    1,       102,        3
                    2,       201,        1
                    2,       201,        2



                data = [
                        {1: {101: [1, 2], 102: [3]}},
                        {2: {201: [1, 2]}}
                    ]


        '''


    ##


    def create_folder(self):


        folder_path = Path(f'{self.save_directory}')

        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)

        

        self.fold_path = Path(f'{self.save_directory}/fold_{self.val_fold}')

        if not self.fold_path.exists():
            self.fold_path.mkdir(parents=True, exist_ok=True)


        self.dataset_path = Path(f'{self.save_directory}/fold_{self.val_fold}/datasets')

        if not self.dataset_path.exists():
            self.dataset_path.mkdir(parents=True, exist_ok=True)       



        folder_path = Path(f'{self.save_directory}/fold_{self.val_fold}/datasets/train')

        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)



        self.train_image_path = Path(f'{self.save_directory}/fold_{self.val_fold}/datasets/train/images')

        if not self.train_image_path.exists():
            self.train_image_path.mkdir(parents=True, exist_ok=True)


        self.train_labels_path = Path(f'{self.save_directory}/fold_{self.val_fold}/datasets/train/labels')

        if not self.train_labels_path.exists():
            self.train_labels_path.mkdir(parents=True, exist_ok=True)
 


        folder_path = Path(f'{self.save_directory}/fold_{self.val_fold}/datasets/val')

        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)


        self.val_images_path = Path(f'{self.save_directory}/fold_{self.val_fold}/datasets/val/images')

        if not self.val_images_path.exists():
            self.val_images_path.mkdir(parents=True, exist_ok=True)



        self.val_labels_path = Path(f'{self.save_directory}/fold_{self.val_fold}/datasets/val/labels')

        if not self.val_labels_path.exists():
            self.val_labels_path.mkdir(parents=True, exist_ok=True)







    def read_cross_validation(self):


        df = pd.read_csv(self.csv_directory)

        # Filter the data for fold 1



        self.validation_data = df[df['fold'] == self.val_fold]

        print('len val',len(self.validation_data))

        self.training_data = df[df['fold'] != self.val_fold]


        print('len train',len(self.training_data))




    def read_dicom(self,dicom_dir):
        ds = pydicom.dcmread(dicom_dir)

        image = ds.pixel_array

        image = (image - image.min()) / (image.max() - image.min() +1e-6) * 255
        image = np.stack([image]*3, axis=-1).astype('uint8')


        return image




    '''

    In dicom to png we convert the dicom data to png and save it and also we calculate the 
    height and weight of image ( for creating the labels)


    '''


    def dicom_to_png(self,df,image_directory):

        self.height_weight_info=[]


        for study_id, study_group in df.groupby('study_id'):




            for series_id, series_group in study_group.groupby('series_id'):


                instances = series_group['instance_number'].tolist()

                unique_instance= np.unique(instances)


           
                ## Find the height and  weight and save it 

                dcm_diretory=f'{self.dataset_directory}/{study_id}/{series_id}/{instances[0]}.dcm'

                dcm_image = self.read_dicom(dcm_diretory)

                height, width, channels = dcm_image.shape


                self.height_weight_info.append({'study_id':study_id , 'series_id': series_id, 'height': height , 'width':width})



                for instance in unique_instance: 
                        
                    dcm_diretory=f'{self.dataset_directory}/{study_id}/{series_id}/{instance}.dcm'

                    dcm_image = self.read_dicom(dcm_diretory)

                    height, width, channels = dcm_image.shape




                    cv2.imwrite(f'./{image_directory}/{study_id}_{series_id}_{instance}.png', dcm_image)


                


    '''
        In this part we save the height and weight of each subject

    '''


    def save_weight_height_to_csv(self):


        csv_file = f'./{self.fold_path}/{self.condition_name}_height_weight.csv'

        # Write the data to a CSV file
        with open(csv_file, mode='w', newline='') as file:
            # Create a CSV DictWriter object
            writer = csv.DictWriter(file, fieldnames=self.height_weight_info[0].keys())
            
            # Write the header (field names)
            writer.writeheader()
            
            # Write the rows (each dictionary)
            writer.writerows(self.height_weight_info)

        print(f"Data saved to {csv_file}")



        # Load the first CSV file
        file2 = pd.read_csv(f'./{self.fold_path}/{self.condition_name}_height_weight.csv')

        # Load the second CSV file
        file1 = pd.read_csv(self.csv_directory)

        # Merge the two DataFrames based on 'study_id' and 'series_id'
        # This will keep all rows from file1 and add 'height' and 'weight' from file2
        merged_data = pd.merge(file1, file2[['study_id', 'series_id', 'height', 'width']],
                            on=['study_id', 'series_id'], how='left')

        # Save the merged data into a new CSV file
        merged_data.to_csv(f'./{self.fold_path}/{self.condition_name}_height_weight.csv', index=False)

        print("Merged data has been saved to 'updated_file.csv'")



    



    def find_class_label(self,condition,level):

        condition=condition.replace(' ','_')
        level=level.replace('/','_')

        condtion_level=f'{condition}_{level}'


        return self.condition_level_classes[condtion_level]










    
    def create_label_for_yolo(self,df,labels_directory):



        df = pd.read_csv(f'./{self.fold_path}/{self.condition_name}_height_weight.csv')


        for study_id, study_group in df.groupby('study_id'):


            for series_id, series_group in study_group.groupby('series_id'):


                for instances_number ,instances_groups in series_group.groupby('instance_number'):

                    labels=[]


                    for (condition, level), group in instances_groups.groupby(['condition', 'level']):

                  

                        x=group['x']

                        y= group['y']

                        height=group['height']
                        
                        width=group['width']


                        condtion_calss=self.find_class_label(condition,level)


                        labels.append({
                        'class_id':condtion_calss,
                        'x': float(x/width),
                        'y': float(y/height),
                        'width': float(self.width_box/width),
                        'height': float(self.width_box/height),
                    
                        })
                    


                    
                    output_file=f'{labels_directory}/{study_id}_{series_id}_{instances_number}.txt'

                    with open(output_file, 'w') as file:
                        for item in labels:

                            class_id = item['class_id']
                            x = item['x']
                            y = item['y']
                            width = item['width']
                            height = item['height']

                            file.write(f"{class_id} {x} {y} {width} {height}\n")


    

    def creata_yaml_file(self):
        
        yaml_file_path = f'{self.dataset_path}/yolo_config.yaml'

        num_classes = len(self.condition_level_classes)

        # Prepare the YAML data
        yaml_data = {
            'train':'./train' ,  # Assuming training images are in JPG format
            'val': './val',    # Assuming validation images are also in JPG format
            'nc': num_classes,
            'names': [name for name in self.condition_level_classes.keys()]
        }
        
        # Write the YAML file
        with open(yaml_file_path, 'w') as file:
            yaml.dump(yaml_data, file, default_flow_style=False)



















                











#### Spinal canal stenosis 


for fold in range(5):

    condition_level_classes_spinal_canal={

        'Spinal_Canal_Stenosis_L1_L2':0,

        'Spinal_Canal_Stenosis_L2_L3':1,

        'Spinal_Canal_Stenosis_L3_L4':2,

        'Spinal_Canal_Stenosis_L4_L5':3,    

        'Spinal_Canal_Stenosis_L5_S1':4,

    }


    d=Detector_data_prepration(
        


        dataset_directory='../../Data/train_images',

        

        csv_directory='../cross_validation/Spinal_Canal_Stenosis_folds.csv',

        condition_name='Spinal_Canal_Stenosis',
        
        
        
        condition_level_classes=condition_level_classes_spinal_canal,


        val_fold=fold,


        width_box=16,



        
        )




print('Spinal canal stenosis  DONE')






#### Subarticular_Stenosis


for fold in range(5):

    condition_level_classes_spinal_canal={

        'Left_Subarticular_Stenosis_L1_L2':0,

        'Left_Subarticular_Stenosis_L2_L3':1,

        'Left_Subarticular_Stenosis_L3_L4':2,

        'Left_Subarticular_Stenosis_L4_L5':3,    

        'Left_Subarticular_Stenosis_L5_S1':4,






        'Right_Subarticular_Stenosis_L1_L2':5,

        'Right_Subarticular_Stenosis_L2_L3':6,

        'Right_Subarticular_Stenosis_L3_L4':7,

        'Right_Subarticular_Stenosis_L4_L5':8,    

        'Right_Subarticular_Stenosis_L5_S1':9,

    }


    d=Detector_data_prepration(
        


        dataset_directory='../../Data/train_images',

        

        csv_directory='../cross_validation/Subarticular_Stenosis_folds.csv',

        condition_name='Subarticular_Stenosis',
        
        
        
        condition_level_classes=condition_level_classes_spinal_canal,


        val_fold=fold,


        width_box=16,



        
        )








print('Subarticular_Stenosis  DONE')




# ####  Neural Foraminal Narrowing



for fold in range(5):

    condition_level_classes_spinal_canal={

        'Left_Neural_Foraminal_Narrowing_L1_L2':0,

        'Left_Neural_Foraminal_Narrowing_L2_L3':1,

        'Left_Neural_Foraminal_Narrowing_L3_L4':2,

        'Left_Neural_Foraminal_Narrowing_L4_L5':3,    

        'Left_Neural_Foraminal_Narrowing_L5_S1':4,






        'Right_Neural_Foraminal_Narrowing_L1_L2':5,

        'Right_Neural_Foraminal_Narrowing_L2_L3':6,

        'Right_Neural_Foraminal_Narrowing_L3_L4':7,

        'Right_Neural_Foraminal_Narrowing_L4_L5':8,    

        'Right_Neural_Foraminal_Narrowing_L5_S1':9,

    }


    d=Detector_data_prepration(
        


        dataset_directory='../../Data/train_images',

        

        csv_directory='../cross_validation/Neural_Foraminal_Narrowing_folds.csv',

        condition_name='Left Neural Foraminal Narrowing',
        
        
        
        condition_level_classes=condition_level_classes_spinal_canal,


        val_fold=fold,


        width_box=16,



        
        )







print(' Neural Foraminal Narrowing  DONE')







