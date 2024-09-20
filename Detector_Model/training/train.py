

from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image

from pathlib import Path
import yaml
#



class yolo_training:


    def __init__(self,
    
        data_directory='',

        condition='',

        fold=0,


        results_directory='',

        epochs=500,

        patience=20,

        batch=4,

    
    ):


        self.condition=condition

        self.fold=fold


        self.results_directory=results_directory

        self.data_directory=data_directory

        self.epochs=epochs
        self.patience=patience
        self.batch=batch


        # the yolo that we need for training

        self.yolo_config_yaml=f'{self.data_directory}/{self.condition}/fold_{self.fold}/datasets/yolo_config.yaml'




        ## Create a folder 


        self.update_yaml()

        model=self.load_pretrain_model()

        self.training(model)








    

    def load_pretrain_model(self):

        model = YOLO('yolov8n.yaml')  # build a new model from YAML

        model = YOLO('yolov8n.pt')

        return model






    '''
        We clear the dataset directory and let the yaml file that will read for training update that

        The resoan is that sone time the yaml file do not let update 

    '''

    def update_yaml(self):


        # Define the path to your settings file
        settings_file = '/users/amousavi/.config/Ultralytics/settings.yaml'

        # Define the new dataset directory
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








        self.result_condition_fold_directory=f'{self.results_directory}/{self.condition}/{self.fold}'



    def training(self,model):

        result_condition_path = Path(f'{self.results_directory}/{self.condition}')

        if not result_condition_path.exists():
            result_condition_path.mkdir(parents=True, exist_ok=True)


        result_condition_fold_path = Path(f'{self.results_directory}/{self.condition}/fold_{self.fold}')

        if not result_condition_fold_path.exists():
            result_condition_fold_path.mkdir(parents=True, exist_ok=True)


        #Define subdirectory for this specific training
        name = "epochs-" 





        results = model.train(data=self.yolo_config_yaml,
                            project=result_condition_fold_path,
                            name=name,
                            epochs=self.epochs,
                            patience=self.patience, #I am setting patience=0 to disable early stopping.
                            batch=self.batch,
                            )






    







            



yolo_training( 
        data_directory=args.data_directory,

        condition=args.condition,

        fold=args.fold,

        results_directory=args.results_directory,

        epochs=args.epochs,

        patience=args.patients,

        batch=args.batch,
    )






import argparse


parser = argparse.ArgumentParser(description='Model Training Parameters')  

parser.add_argument('--data_directory', type=str, help='data that prepare for train directory')

parser.add_argument('--condition', type=str, help='condition')

parser.add_argument('--results_directory', type=str, help='Results directory')

parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')

parser.add_argument('--fold', type=int, default=0, help='the fold that you want to train')

parser.add_argument('--patients', type=int, default=5, help='Number of patients')

parser.add_argument('--batch', type=int, default=4, help='Batch size')

args=parser.parse_args()





# python3 train.py --data_directory  ../data_prepration  --condition Spinal_Canal_Stenosis  --epochs 500  --fold 0 --patients 20 --batch 4
