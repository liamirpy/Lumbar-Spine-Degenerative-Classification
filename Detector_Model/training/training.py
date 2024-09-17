

from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image

from pathlib import Path

model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')





### make sure the yaml file is define

import yaml

# Define the path to your settings file
settings_file = '/users/amousavi/.config/Ultralytics/settings.yaml'

# Define the new dataset directory
# new_dataset_dir = '/users/amousavi/Challenge/Lumbar-Spine-Degenerative-Classification/Detector_Model/data_prepration/Spinal_Canal_Stenosis/fold_1/datasets'
new_dataset_dir=''

# Load the YAML file
with open(settings_file, 'r') as file:
    settings = yaml.safe_load(file)

# Update the dataset directory in the YAML configuration
settings['datasets_dir'] = new_dataset_dir

# Save the updated YAML back to the file
with open(settings_file, 'w') as file:
    yaml.dump(settings, file, default_flow_style=False)

print(f"Dataset directory has been updated to: {new_dataset_dir}")








result_path = "../results"






def training(condition,fold,epochs):



    result_condition_path = Path(f'{result_path}/condition')

    if not result_condition_path.exists():
        result_condition_path.mkdir(parents=True, exist_ok=True)


    result_condition_fold_path = Path(f'{result_path}/condition/fold_{fold}')

    if not result_condition_fold_path.exists():
        result_condition_fold_path.mkdir(parents=True, exist_ok=True)


    #Define subdirectory for this specific training
    name = "epochs-" #note that if you run the training again, it creates a directory: 200_epochs-2





    results = model.train(data=f'../data_prepration/{condition}/fold_{fold}/datasets/yolo_config.yaml',
                        project=result_condition_fold_path,
                        name=name,
                        epochs=epochs,
                        patience=0, #I am setting patience=0 to disable early stopping.
                        batch=4,
                        )






epochs=3



training('Spinal_Canal_Stenosis','1',epochs)