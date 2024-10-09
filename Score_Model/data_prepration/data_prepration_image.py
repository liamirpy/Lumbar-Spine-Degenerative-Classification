'''
In this code we convert dcm to png for each condition and its folds


'''

import pydicom

import pandas as pd 
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import numpy as np



class data_prepration:
    def __init__(self,
    
        dataset_directory='',

        condition='' ,

        csv_directory='',

        num_fold=5,


        augmentation_list=['rotate', 'horizontal_flip','vertical_flip', 'gaussian_noise']

        
    
    
    ):


        self.dataset_directory=dataset_directory

        self.condition=condition

        print('Data prepration for :',self.condition)

        self.augmentation_list=augmentation_list

        self.number_of_fold=num_fold

        self.csv_directory=csv_directory


        ##  create a folder 
        self.create_folder()


        self.dicom_to_png_all_fold()












    
    def create_folder(self):



        folder_path = Path(f'./{self.condition}')

        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)




        print(self.number_of_fold)
        for fold in range(self.number_of_fold):

            self.fold_path = Path(f'./{self.condition}/fold_{fold}')

            if not self.fold_path.exists():
                self.fold_path.mkdir(parents=True, exist_ok=True)



            self.fold_path = Path(f'./{self.condition}/fold_{fold}/train')

            if not self.fold_path.exists():
                self.fold_path.mkdir(parents=True, exist_ok=True)



            self.fold_path = Path(f'./{self.condition}/fold_{fold}/val')

            if not self.fold_path.exists():
                self.fold_path.mkdir(parents=True, exist_ok=True)




    def read_csv(self,fold,train_or_val):


        if train_or_val == 'train':


            df = pd.read_csv(f'{self.csv_directory}/{self.condition}/fold_{fold}/{self.condition}_augmented.csv')

        
        if train_or_val == 'val':

            df = pd.read_csv(f'{self.csv_directory}/{self.condition}/fold_{fold}/{self.condition}_val.csv')



        return df

    







    def read_dicom(self,dicom_dir):


        ds = pydicom.dcmread(dicom_dir)

        image = ds.pixel_array

        image = (image - image.min()) / (image.max() - image.min() +1e-6) * 255
        image = np.stack([image]*3, axis=-1).astype('uint8')


        return image


    

    def crop_(self,image,x,y):


        img = Image.fromarray(image)
        
        # Define the crop box
        left = int(x - 16)
        top = int(y - 16)
        right = int(x + 16)
        bottom = int(y + 16)
        crop_box = (left, top, right, bottom)

        cropped_img = img.crop(crop_box)

        return cropped_img


    


    def rotate_image(self,image: Image.Image, angle: float) -> Image.Image:
    
        return image.rotate(angle, expand=True)


    def horizontal_flip(self,image: Image.Image) -> Image.Image:
        """
        Flip the image horizontally.
        """
        return ImageOps.mirror(image)

    def vertical_flip(self,image: Image.Image) -> Image.Image:
        """
        Flip the image vertically.
        """
        return ImageOps.flip(image)

    def add_gaussian_noise(self,image: Image.Image, mean=0, std=25) -> Image.Image:
        """
        Add Gaussian noise to the image.
        """
        np_image = np.array(image)
        
        # Generate Gaussian noise
        noise = np.random.normal(mean, std, np_image.shape)
        
        # Add the noise to the image and clip to keep pixel values within valid range
        noisy_image = np_image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_image) 





    


    def dicom_to_png_all_fold(self):



        for fold in range(self.number_of_fold):

            print('fold',fold)


            df=self.read_csv(fold,'train')


            for index, row in df.iterrows():

                study_id = str(row['study_id'])

                series_id = str(row['series_id'])

                instance_number = str(row['instance_number'])

                x = row['x']

                int_x=str(int(x))

                y = row['y']

                int_y=str(int(y))

                augmentation = str(row['augmentation'])



                dcm_diretory=f'{self.dataset_directory}/{study_id}/{series_id}/{instance_number}.dcm'

                dcm_image = self.read_dicom(dcm_diretory)


                cropped_img= self.crop_(dcm_image,x,y)



                if augmentation in self.augmentation_list:

                    if augmentation=='rotate':


                        angle=np.random.uniform(1, 20)


                        augmented_images=self.rotate_image(cropped_img,angle)

                        augmented_images.save(f'./{self.condition}/fold_{fold}/train/{study_id}_{series_id}_{instance_number}_{int_x}_{int_y}_augmented.png')



                    if augmentation=='horizontal_flip':


                        augmented_images=self.horizontal_flip(cropped_img)
                        augmented_images.save(f'./{self.condition}/fold_{fold}/train/{study_id}_{series_id}_{instance_number}_{int_x}_{int_y}_augmented.png')



                    if augmentation=='vertical_flip':


                        augmented_images=self.vertical_flip(cropped_img)
                        augmented_images.save(f'./{self.condition}/fold_{fold}/train/{study_id}_{series_id}_{instance_number}_{int_x}_{int_y}_augmented.png')


                    if augmentation=='gaussian_noise':

                        augmented_images=self.add_gaussian_noise(cropped_img)
                        augmented_images.save(f'./{self.condition}/fold_{fold}/train/{study_id}_{series_id}_{instance_number}_{int_x}_{int_y}_augmented.png')

        
                else:


                    cropped_img.save(f'./{self.condition}/fold_{fold}/train/{study_id}_{series_id}_{instance_number}_{int_x}_{int_y}.png')





            df=self.read_csv(fold,'val')


            for index, row in df.iterrows():

                study_id = str(row['study_id'])

                series_id = str(row['series_id'])

                instance_number = str(row['instance_number'])

                x = row['x']

                int_x=str(int(x))

                y = row['y']

                int_y=str(int(y))


                dcm_diretory=f'{self.dataset_directory}/{study_id}/{series_id}/{instance_number}.dcm'

                dcm_image = self.read_dicom(dcm_diretory)


                cropped_img= self.crop_(dcm_image,x,y)

                cropped_img.save(f'./{self.condition}/fold_{fold}/val/{study_id}_{series_id}_{instance_number}_{int_x}_{int_y}.png')



                










data_prepration(
    
    
    
    
    dataset_directory='../../Data/train_images',
    
    
    condition='Subarticular_Stenosis',


    csv_directory='../data_augmentation/augmented_output',
    
    num_fold=5,

    augmentation_list=['rotate', 'horizontal_flip','vertical_flip', 'gaussian_noise']

    )








data_prepration(
    
    
    
    
    dataset_directory='../../Data/train_images',
    
    
    condition='Spinal_Canal_Stenosis',


    csv_directory='../data_augmentation/augmented_output',
    
    num_fold=5,

    augmentation_list=['rotate', 'horizontal_flip','vertical_flip', 'gaussian_noise']

    )








data_prepration(
    
    
    
    
    dataset_directory='../../Data/train_images',
    
    
    condition='Neural_Foraminal_Narrowing',


    csv_directory='../data_augmentation/augmented_output',
    
    num_fold=5,

    augmentation_list=['rotate', 'horizontal_flip','vertical_flip', 'gaussian_noise']

    )





