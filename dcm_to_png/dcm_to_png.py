'''

Converting dcm to png for each csv (3 condition) 


'''




import os
import pandas as pd
import pydicom
from PIL import Image
import numpy as np







def coonvert_dcm_to_png(csv_directory):
# Load the CSV file
    csv_file = f'../Data_Splitting/{csv_directory}.csv'

    data_directory='../Data/train_images'
    df = pd.read_csv(csv_file)

    # Create a new directory for PNG images
    output_dir = "./Neural_Foraminal_Narrowing_png"
    os.makedirs(output_dir, exist_ok=True)

    def normalize_image(image_array):
        """Normalize pixel values to the range 0-255"""
        img_min = np.min(image_array)
        img_max = np.max(image_array)
        if img_max > img_min:
            normalized_array = 255 * (image_array - img_min) / (img_max - img_min)
        else:
            normalized_array = np.zeros_like(image_array)
        return normalized_array.astype(np.uint8)

    for _, row in df.iterrows():
        study_id = row['study_id']
        series_id = row['series_id']

        # Create directory structure
        series_dir = os.path.join(output_dir, str(study_id), str(series_id))
        os.makedirs(series_dir, exist_ok=True)

        # Path to DICOM files (assuming .dcm files are in a subdirectory)
        dicom_dir = os.path.join(data_directory,str(study_id), str(series_id))

        # Convert each DICOM file to PNG
        for dicom_file in os.listdir(dicom_dir):
            if dicom_file.endswith('.dcm'):
                dicom_path = os.path.join(dicom_dir, dicom_file)
                dicom_data = pydicom.dcmread(dicom_path)
                
                # Normalize the DICOM image
                img_array = dicom_data.pixel_array
                img_array = normalize_image(img_array)
                img = Image.fromarray(img_array)
                
                # Save the PNG image
                png_file = os.path.splitext(dicom_file)[0] + '.png'
                png_path = os.path.join(series_dir, png_file)
                img.save(png_path)

    print("Conversion and normalization completed.")




coonvert_dcm_to_png('Neural_Foraminal_Narrowing')
coonvert_dcm_to_png('Subarticular_Stenosis')
coonvert_dcm_to_png('Spinal_Canal_Stenosis')
