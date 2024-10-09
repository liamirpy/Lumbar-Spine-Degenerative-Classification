
'''

Prediction model for test data



the test csv file should be like this :



        study_id,series_id,series_description

        44036939,2828203845,Sagittal T1

        44036939,3481971518,Axial T2

        44036939,3844393089,Sagittal T2/STIR



For this datast for each modality we have a single condition 


    # - Neural Foraminal Narrowing: Sagittal T1

    # - Spinal Canal Stenosis: Sagittal T2

    # - Subarticular Stenosis: Axial T2 






'''


from tensorflow.keras.preprocessing.image import load_img, img_to_array

from ultralytics import YOLO
from tensorflow import keras
import pandas as pd
import numpy as np
import pydicom
import os 
from PIL import Image, ImageOps
import cv2
import csv





class predictior:

    def __init__(self,

    test_csv='',

    test_images='',

    model_path='',
    
    ):



        self.test_csv=test_csv

        self.test_images=test_images

        self.model_path=model_path

        results=self.prediction()

        self.combining_results(results)







    def loss_function(self,gamma=2., alpha=None, class_weights=None):
        """
        Focal Loss function for binary and multi-class classification tasks with class weights.
        
        Args:
        - gamma: focusing parameter for modulating factor (1-p)
        - alpha: balancing factor for each class. Can be a scalar (binary) or a list (multi-class).
        - class_weights: A list or array of class weights to handle class imbalance
        
        Returns:
        - loss: computed focal loss value
        """
        
        def focal_loss(y_true, y_pred):
            # Clip predictions to avoid log(0) errors
            y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

            # Cross-entropy loss
            cross_entropy_loss = -y_true * K.log(y_pred)
            
            # Modulating factor (1 - p)^gamma
            modulating_factor = K.pow(1.0 - y_pred, gamma)
            
            # Apply class weights if provided
            if class_weights is not None:
                class_weights_tensor = K.constant(class_weights, dtype=tf.float32)
                weights = y_true * class_weights_tensor
                cross_entropy_loss *= weights

            # Apply alpha if provided (handling binary and multi-class cases)
            if alpha is not None:
                if isinstance(alpha, (float, int)):  # Binary classification
                    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
                else:  # Multi-class classification
                    alpha_tensor = K.constant(alpha, dtype=tf.float32)
                    alpha_factor = y_true * alpha_tensor
                cross_entropy_loss *= alpha_factor
            
            # Final focal loss computation
            loss = modulating_factor * cross_entropy_loss
            return K.sum(loss, axis=-1)

        return focal_loss



    def find_condition_based_on_modality(self,modality):


        modality_condition={'Sagittal_T1':'Neural_Foraminal_Narrowing',
                            'Sagittal_T2/STIR':'Spinal_Canal_Stenosis',
                            'Axial_T2':'Subarticular_Stenosis'}

        return modality_condition[modality]




    
    def dicom_files_list(self,folder_path):
        dicom_filenames = []
        
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".dcm"):  # Check if the file is a DICOM file
                dicom_filenames.append(filename)
        
        return dicom_filenames
        


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








    def Neural_Foraminal_Narrowing(self,study_id,series_id,prediction):

        print('The General condition is Neural Foraminal Narrowing')


        prediction_result=prediction



        list_of_dcm_file=self.dicom_files_list(f'{self.test_images}/{study_id}/{series_id}')



        condition_detector_model = YOLO(f"{self.model_path}/condtion_detector/Neural_Foraminal_Narrowing.pt")


        score_prediction_model = keras.models.load_model(f"{self.model_path}/score_predictor/Neural_Foraminal_Narrowing.keras",
        
        custom_objects={'focal_loss':self.loss_function})



        for dcm in list_of_dcm_file:

            dcm_as_image=self.read_dicom(f'{self.test_images}/{study_id}/{series_id}/{dcm}')

            # cv2.imwrite(f'./image_1.png', dcm_as_image)

            # cv2.imwrite('./kk.png', dcm_as_image)
            # dcm_as_image=cv2.imread('./kk.png')

            results = condition_detector_model(dcm_as_image)  # predict on an image


            result=results[0]

            boxes=result.boxes

            classes_dictionary=result.names


            # print('classes_dictionary:',classes_dictionary)



            # print('Number of box:',len(boxes))




            if len(boxes) !=0:
                    
                for box in boxes:

                    belong_to_class=box.cls[0].item()

                    confidential=box.conf[0].item()

                    box_cordination=box.xywh[0].tolist()

                    x_center=box_cordination[0]

                    y_center=box_cordination[1]


                    # print('Belong to Class:',belong_to_class, 
                    
                    #         'With confidential:',confidential,'box cordination',box_cordination)
                    

                    croped_image=self.crop_(dcm_as_image,x_center,y_center)

                    croped_image.save('./image.png')

                    img_list = []
                    img = load_img('./image.png', target_size=(32, 32))  # Resize to 224x224
                    img_array = img_to_array(img)
                    img_list.append(img_array)

                    image_array_for_prediction=np.array(img_list)





                    predictions = score_prediction_model.predict(image_array_for_prediction)

                    # print( np.array((predictions[0])) )


                    prediction_result.append({'condition':classes_dictionary[belong_to_class],

                                                'confident_of_condition':confidential,

                                                'score_prediction':list((predictions[0])) })




        return prediction_result













    def Spinal_Canal_Stenosis(self,study_id,series_id,prediction):

        print('The General condition is Spinal Canal Stenosis')


        prediction_result=prediction






        list_of_dcm_file=self.dicom_files_list(f'{self.test_images}/{study_id}/{series_id}')



        condition_detector_model = YOLO(f"{self.model_path}/condtion_detector/Spinal_Canal_Stenosis.pt")


        score_prediction_model = keras.models.load_model(f"{self.model_path}/score_predictor/Spinal_Canal_Stenosis.keras",
        
        custom_objects={'focal_loss':self.loss_function})



        for dcm in list_of_dcm_file:

            dcm_as_image=self.read_dicom(f'{self.test_images}/{study_id}/{series_id}/{dcm}')


            # cv2.imwrite('./kk.png', dcm_as_image)

            results = condition_detector_model(dcm_as_image)  # predict on an image


            result=results[0]

            boxes=result.boxes

            classes_dictionary=result.names


            # print('classes_dictionary:',classes_dictionary)



            # print('Number of box:',len(boxes))




            if len(boxes) !=0:
                    
                for box in boxes:

                    belong_to_class=box.cls[0].item()

                    confidential=box.conf[0].item()

                    box_cordination=box.xywh[0].tolist()

                    x_center=box_cordination[0]

                    y_center=box_cordination[1]


                    # print('Belong to Class:',belong_to_class, 
                    
                    #         'With confidential:',confidential,'box cordination',box_cordination)
                    

                    croped_image=self.crop_(dcm_as_image,x_center,y_center)

                    croped_image.save('./image.png')

                    img_list = []
                    img = load_img('./image.png', target_size=(32, 32))  # Resize to 224x224
                    img_array = img_to_array(img)
                    img_list.append(img_array)

                    image_array_for_prediction=np.array(img_list)





                    predictions = score_prediction_model.predict(image_array_for_prediction)

                    # print( np.array((predictions[0])) )


                    prediction_result.append({'condition':classes_dictionary[belong_to_class],

                                                'confident_of_condition':confidential,

                                                'score_prediction':list((predictions[0])) })




        return prediction_result








    def Subarticular_Stenosis(self,study_id,series_id,prediction):

        print('The General condition is Subarticular Stenosis')


        prediction_result=prediction






        list_of_dcm_file=self.dicom_files_list(f'{self.test_images}/{study_id}/{series_id}')



        condition_detector_model = YOLO(f"{self.model_path}/condtion_detector/Subarticular_Stenosis.pt")


        score_prediction_model = keras.models.load_model(f"{self.model_path}/score_predictor/Subarticular_Stenosis.keras",
        
        custom_objects={'focal_loss':self.loss_function})



        for dcm in list_of_dcm_file:

            dcm_as_image=self.read_dicom(f'{self.test_images}/{study_id}/{series_id}/{dcm}')


            # cv2.imwrite('./kk.png', dcm_as_image)

            results = condition_detector_model(dcm_as_image)  # predict on an image


            result=results[0]

            boxes=result.boxes

            classes_dictionary=result.names


            # print('classes_dictionary:',classes_dictionary)



            # print('Number of box:',len(boxes))




            if len(boxes) !=0:
                    
                for box in boxes:

                    belong_to_class=box.cls[0].item()

                    confidential=box.conf[0].item()

                    box_cordination=box.xywh[0].tolist()

                    x_center=box_cordination[0]

                    y_center=box_cordination[1]


                    # print('Belong to Class:',belong_to_class, 
                    
                    #         'With confidential:',confidential,'box cordination',box_cordination)
                    

                    croped_image=self.crop_(dcm_as_image,x_center,y_center)

                    croped_image.save('./image.png')

                    img_list = []
                    img = load_img('./image.png', target_size=(32, 32))  # Resize to 224x224
                    img_array = img_to_array(img)
                    img_list.append(img_array)

                    image_array_for_prediction=np.array(img_list)





                    predictions = score_prediction_model.predict(image_array_for_prediction)

                    # print( np.array((predictions[0])) )


                    prediction_result.append({'condition':classes_dictionary[belong_to_class],

                                                'confident_of_condition':confidential,

                                                'score_prediction':list((predictions[0])) })




        return prediction_result




    
    






    def prediction(self):

        '''
            The structure of result will fill like this 

        # d={'study_id_series_id':{'modality':'','condition':'','prediction':[]}}

        '''
        results={}

        df = pd.read_csv(f'{self.test_csv}')

        for index, row in df.iterrows():


            study_id=row['study_id']

            series_id=row['series_id']



            modality=row['series_description'].replace(' ','_')

            condition=self.find_condition_based_on_modality(modality)


            study_id_series_id= study_id+series_id

            if study_id_series_id not in list(results.keys()):

                results[f'si_{study_id}_sei_{series_id}']={'modality':f'{modality}','condition':f'{condition}','prediction':[]}



            '''
                The structure of prediction will fill like this 

            # 'prediction':[{'condition':'','confident_of_condition':int,'score_prediction':[int,int,int]}]

            '''


            if condition == 'Neural_Foraminal_Narrowing':

                
                prediction_result=self.Neural_Foraminal_Narrowing(study_id,series_id,results[f'si_{study_id}_sei_{series_id}']['prediction'])
                results[f'si_{study_id}_sei_{series_id}']['prediction']=prediction_result




            elif condition== 'Spinal_Canal_Stenosis':

                prediction_result=self.Spinal_Canal_Stenosis(study_id,series_id,results[f'si_{study_id}_sei_{series_id}']['prediction'])
                results[f'si_{study_id}_sei_{series_id}']['prediction']=prediction_result


                
            elif condition == 'Subarticular_Stenosis':

                prediction_result=self.Subarticular_Stenosis(study_id,series_id,results[f'si_{study_id}_sei_{series_id}']['prediction'])

                results[f'si_{study_id}_sei_{series_id}']['prediction']=prediction_result


        

        print(results)


        return results

        




    def combining_results(self,result):


        '''
            Report for all study id (in this test data we only have single study id)

            study_ids_report={'study_id':
            
            
            {


                'Left_Subarticular_Stenosis_L1_L2':[],

                'Left_Subarticular_Stenosis_L2_L3':[],

                'Left_Subarticular_Stenosis_L3_L4':[],

                'Left_Subarticular_Stenosis_L4_L5':[],    

                'Left_Subarticular_Stenosis_L5_S1':[],

                'Right_Subarticular_Stenosis_L1_L2':[],

                'Right_Subarticular_Stenosis_L2_L3':[],

                'Right_Subarticular_Stenosis_L3_L4':[],

                'Right_Subarticular_Stenosis_L4_L5':[],    

                'Right_Subarticular_Stenosis_L5_S1':[],





                'Left_Neural_Foraminal_Narrowing_L1_L2':[],

                'Left_Neural_Foraminal_Narrowing_L2_L3':[],

                'Left_Neural_Foraminal_Narrowing_L3_L4':[],

                'Left_Neural_Foraminal_Narrowing_L4_L5':[],    

                'Left_Neural_Foraminal_Narrowing_L5_S1':[],

                'Right_Neural_Foraminal_Narrowing_L1_L2':[],

                'Right_Neural_Foraminal_Narrowing_L2_L3':[],

                'Right_Neural_Foraminal_Narrowing_L3_L4':[],

                'Right_Neural_Foraminal_Narrowing_L4_L5':[],    

                'Right_Neural_Foraminal_Narrowing_L5_S1':[],






                'Spinal_Canal_Stenosis_L1_L2':[],

                'Spinal_Canal_Stenosis_L2_L3':[],

                'Spinal_Canal_Stenosis_L3_L4':[],

                'Spinal_Canal_Stenosis_L4_L5':[],    

                'Spinal_Canal_Stenosis_L5_S1':[],

                
            
            
            
            
            }
            
        
            }

        
        Then we add the result to the list for each condition( it could be more than on detected same condtion)
        we calculate the mean 

        '''


        # result={'si_44036939_sei_2828203845':
        #  {'modality': 'Sagittal_T1', 'condition': 'Neural_Foraminal_Narrowing',
        #   'prediction': [{'condition': 'Left_Neural_Foraminal_Narrowing_L3_L4', 'confident_of_condition': 0.33207300305366516, 'score_prediction': [0.06646155, 0.34472957, 0.5888089]}, {'condition': 'Left_Neural_Foraminal_Narrowing_L4_L5', 'confident_of_condition': 0.2544807493686676, 'score_prediction': [0.2614578, 0.5653935, 0.17314874]}, {'condition': 'Left_Neural_Foraminal_Narrowing_L5_S1', 'confident_of_condition': 0.30506473779678345, 'score_prediction': [0.385176, 0.5175311, 0.09729294]}, {'condition': 'Right_Neural_Foraminal_Narrowing_L4_L5', 'confident_of_condition': 0.2558341324329376, 'score_prediction': [0.51865935, 0.42116457, 0.060176063]}, {'condition': 'Right_Neural_Foraminal_Narrowing_L5_S1', 'confident_of_condition': 0.25508373975753784, 'score_prediction': [0.56941146, 0.33324042, 0.097348176]}]}, 'si_44036939_sei_3481971518': {'modality': 'Axial_T2', 'condition': 'Subarticular_Stenosis', 'prediction': [{'condition': 'Right_Subarticular_Stenosis_L4_L5', 'confident_of_condition': 0.477896124124527, 'score_prediction': [0.49262252, 0.38183156, 0.12554592]}, {'condition': 'Left_Subarticular_Stenosis_L3_L4', 'confident_of_condition': 0.38550207018852234, 'score_prediction': [0.43169117, 0.45046154, 0.11784727]}, {'condition': 'Right_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.2567550837993622, 'score_prediction': [0.5689144, 0.3835391, 0.047546513]}, {'condition': 'Right_Subarticular_Stenosis_L5_S1', 'confident_of_condition': 0.5985532999038696, 'score_prediction': [0.6596228, 0.3030851, 0.037292134]}, {'condition': 'Left_Subarticular_Stenosis_L5_S1', 'confident_of_condition': 0.4833911657333374, 'score_prediction': [0.5323766, 0.41346958, 0.054153785]}, {'condition': 'Left_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.5265852808952332, 'score_prediction': [0.5580876, 0.42469987, 0.017212477]}, {'condition': 'Right_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.30948442220687866, 'score_prediction': [0.6349444, 0.351556, 0.013499647]}, {'condition': 'Left_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.28314119577407837, 'score_prediction': [0.54367, 0.33097824, 0.12535182]}, {'condition': 'Left_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.2506023645401001, 'score_prediction': [0.2836033, 0.5252076, 0.1911891]}, {'condition': 'Left_Subarticular_Stenosis_L3_L4', 'confident_of_condition': 0.31786486506462097, 'score_prediction': [0.027762957, 0.39517483, 0.5770622]}, {'condition': 'Left_Subarticular_Stenosis_L3_L4', 'confident_of_condition': 0.5419588685035706, 'score_prediction': [0.5967957, 0.35130456, 0.051899854]}, {'condition': 'Left_Subarticular_Stenosis_L5_S1', 'confident_of_condition': 0.4901745021343231, 'score_prediction': [0.56190586, 0.38677618, 0.05131801]}, {'condition': 'Left_Subarticular_Stenosis_L3_L4', 'confident_of_condition': 0.4412180185317993, 'score_prediction': [0.07442579, 0.40550673, 0.5200675]}, {'condition': 'Right_Subarticular_Stenosis_L3_L4', 'confident_of_condition': 0.2657001316547394, 'score_prediction': [0.61107093, 0.32859448, 0.06033461]}, {'condition': 'Left_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.38434311747550964, 'score_prediction': [0.61048806, 0.32136452, 0.068147466]}, {'condition': 'Right_Subarticular_Stenosis_L5_S1', 'confident_of_condition': 0.39076027274131775, 'score_prediction': [0.38723224, 0.4727307, 0.14003712]}, {'condition': 'Right_Subarticular_Stenosis_L4_L5', 'confident_of_condition': 0.29849815368652344, 'score_prediction': [0.34832138, 0.4428846, 0.20879397]}, {'condition': 'Left_Subarticular_Stenosis_L2_L3', 'confident_of_condition': 0.42074406147003174, 'score_prediction': [0.4934402, 0.3835948, 0.12296503]}, {'condition': 'Right_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.471079021692276, 'score_prediction': [0.21643329, 0.5539855, 0.22958127]}, {'condition': 'Left_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.36258718371391296, 'score_prediction': [0.58265406, 0.3721228, 0.045223128]}, {'condition': 'Left_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.4119408130645752, 'score_prediction': [0.30227992, 0.4747069, 0.22301327]}, {'condition': 'Right_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.2606167197227478, 'score_prediction': [0.6573978, 0.3296752, 0.01292695]}, {'condition': 'Left_Subarticular_Stenosis_L3_L4', 'confident_of_condition': 0.392037570476532, 'score_prediction': [0.098117985, 0.4587663, 0.44311577]}, {'condition': 'Right_Subarticular_Stenosis_L3_L4', 'confident_of_condition': 0.2912525534629822, 'score_prediction': [0.3077839, 0.4448299, 0.24738611]}, {'condition': 'Right_Subarticular_Stenosis_L4_L5', 'confident_of_condition': 0.3783451318740845, 'score_prediction': [0.30106613, 0.40680933, 0.2921245]}, {'condition': 'Left_Subarticular_Stenosis_L4_L5', 'confident_of_condition': 0.33304157853126526, 'score_prediction': [0.11126637, 0.41072077, 0.47801277]}, {'condition': 'Right_Subarticular_Stenosis_L5_S1', 'confident_of_condition': 0.493558794260025, 'score_prediction': [0.57841, 0.331464, 0.090126015]}, {'condition': 'Right_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.5354706645011902, 'score_prediction': [0.35150337, 0.44609445, 0.20240226]}, {'condition': 'Left_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.31742244958877563, 'score_prediction': [0.21272701, 0.49448532, 0.2927877]}, {'condition': 'Left_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.2804158926010132, 'score_prediction': [0.5764287, 0.38313386, 0.040437385]}, {'condition': 'Left_Subarticular_Stenosis_L2_L3', 'confident_of_condition': 0.7354334592819214, 'score_prediction': [0.090543486, 0.45112512, 0.45833147]}, {'condition': 'Left_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.402029812335968, 'score_prediction': [0.58114105, 0.3865698, 0.032289166]}, {'condition': 'Left_Subarticular_Stenosis_L4_L5', 'confident_of_condition': 0.5458420515060425, 'score_prediction': [0.04505941, 0.41873145, 0.5362091]}, {'condition': 'Right_Subarticular_Stenosis_L4_L5', 'confident_of_condition': 0.4013291597366333, 'score_prediction': [0.1470937, 0.4584519, 0.3944544]}, {'condition': 'Left_Subarticular_Stenosis_L4_L5', 'confident_of_condition': 0.29595330357551575, 'score_prediction': [0.62924004, 0.33599943, 0.034760498]}, {'condition': 'Left_Subarticular_Stenosis_L2_L3', 'confident_of_condition': 0.27071458101272583, 'score_prediction': [0.29003248, 0.513375, 0.1965925]}, {'condition': 'Left_Subarticular_Stenosis_L5_S1', 'confident_of_condition': 0.29917216300964355, 'score_prediction': [0.6008876, 0.3585518, 0.040560607]}, {'condition': 'Right_Subarticular_Stenosis_L5_S1', 'confident_of_condition': 0.4749331772327423, 'score_prediction': [0.3052655, 0.46921295, 0.2255216]}, {'condition': 'Left_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.3490436375141144, 'score_prediction': [0.6310535, 0.34976777, 0.019178815]}, {'condition': 'Right_Subarticular_Stenosis_L2_L3', 'confident_of_condition': 0.6072553992271423, 'score_prediction': [0.50544167, 0.37713793, 0.11742041]}, {'condition': 'Left_Subarticular_Stenosis_L4_L5', 'confident_of_condition': 0.5536304116249084, 'score_prediction': [0.511769, 0.38140875, 0.10682234]}, {'condition': 'Left_Subarticular_Stenosis_L3_L4', 'confident_of_condition': 0.35776713490486145, 'score_prediction': [0.47312236, 0.42407498, 0.10280268]}, {'condition': 'Left_Subarticular_Stenosis_L1_L2', 'confident_of_condition': 0.356276273727417, 'score_prediction': [0.61594814, 0.35687187, 0.027180014]}]}, 'si_44036939_sei_3844393089': {'modality': 'Sagittal_T2/STIR', 'condition': 'Spinal_Canal_Stenosis', 'prediction': [{'condition': 'Spinal_Canal_Stenosis_L4_L5', 'confident_of_condition': 0.4159785807132721, 'score_prediction': [0.7520976, 0.20860788, 0.039294507]}, {'condition': 'Spinal_Canal_Stenosis_L3_L4', 'confident_of_condition': 0.3101487159729004, 'score_prediction': [0.68033665, 0.2841096, 0.035553787]}, {'condition': 'Spinal_Canal_Stenosis_L1_L2', 'confident_of_condition': 0.3923245370388031, 'score_prediction': [0.39756748, 0.49233958, 0.11009299]}, {'condition': 'Spinal_Canal_Stenosis_L4_L5', 'confident_of_condition': 0.38994327187538147, 'score_prediction': [0.517364, 0.39761233, 0.08502363]}, {'condition': 'Spinal_Canal_Stenosis_L3_L4', 'confident_of_condition': 0.375446617603302, 'score_prediction': [0.81694835, 0.16379897, 0.01925265]}, {'condition': 'Spinal_Canal_Stenosis_L4_L5', 'confident_of_condition': 0.3875669836997986, 'score_prediction': [0.33818278, 0.55536956, 0.106447734]}, {'condition': 'Spinal_Canal_Stenosis_L5_S1', 'confident_of_condition': 0.3711393177509308, 'score_prediction': [0.43357658, 0.49678394, 0.069639534]}, {'condition': 'Spinal_Canal_Stenosis_L3_L4', 'confident_of_condition': 0.308648943901062, 'score_prediction': [0.53562, 0.3974621, 0.06691796]}, {'condition': 'Spinal_Canal_Stenosis_L5_S1', 'confident_of_condition': 0.4563785195350647, 'score_prediction': [0.35786095, 0.5057585, 0.13638051]}, {'condition': 'Spinal_Canal_Stenosis_L2_L3', 'confident_of_condition': 0.25070762634277344, 'score_prediction': [0.07301798, 0.37653658, 0.55044544]}, {'condition': 'Spinal_Canal_Stenosis_L1_L2', 'confident_of_condition': 0.3680715560913086, 'score_prediction': [0.5800507, 0.34366578, 0.07628354]}, {'condition': 'Spinal_Canal_Stenosis_L4_L5', 'confident_of_condition': 0.42469388246536255, 'score_prediction': [0.035284687, 0.34921, 0.61550534]}, {'condition': 'Spinal_Canal_Stenosis_L5_S1', 'confident_of_condition': 0.4191510081291199, 'score_prediction': [0.612144, 0.3619612, 0.025894793]}, {'condition': 'Spinal_Canal_Stenosis_L3_L4', 'confident_of_condition': 0.2665574848651886, 'score_prediction': [0.14187671, 0.43903297, 0.41909033]}, {'condition': 'Spinal_Canal_Stenosis_L2_L3', 'confident_of_condition': 0.3390381634235382, 'score_prediction': [0.2518272, 0.38235074, 0.365822]}, {'condition': 'Spinal_Canal_Stenosis_L1_L2', 'confident_of_condition': 0.2614451050758362, 'score_prediction': [0.80600464, 0.1801197, 0.013875708]}]}}




        df = pd.read_csv(f'{self.test_csv}')


        csv_file_path = 'results.csv'

# Open the CSV file for writing
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header
            header = ['ID', 'Modality', 'Condition', 'Predicted Condition', 'Confidence', 'Score Prediction 1', 'Score Prediction 2', 'Score Prediction 3']
            writer.writerow(header)
            
            # Iterate through the dictionary to write the data
            for id_key, details in result.items():
                modality = details['modality']
                condition = details['condition']
                predictions = details['prediction']
                
                for prediction in predictions:
                    predicted_condition = prediction['condition']
                    confidence = prediction['confident_of_condition']
                    score_predictions = prediction['score_prediction']
                    
                    # Write each row to the CSV file
                    writer.writerow([id_key, modality, condition, predicted_condition, confidence] + score_predictions)



        study_ids_report={}
        ### create the study_id_report 

        for index, row in df.iterrows():



            study_id=row['study_id']


            if str(study_id) not in list(study_ids_report.keys()):


                study_ids_report[f'{study_id}']= {


                'Left_Subarticular_Stenosis_L1_L2':[],

                'Left_Subarticular_Stenosis_L2_L3':[],

                'Left_Subarticular_Stenosis_L3_L4':[],

                'Left_Subarticular_Stenosis_L4_L5':[],    

                'Left_Subarticular_Stenosis_L5_S1':[],

                'Right_Subarticular_Stenosis_L1_L2':[],

                'Right_Subarticular_Stenosis_L2_L3':[],

                'Right_Subarticular_Stenosis_L3_L4':[],

                'Right_Subarticular_Stenosis_L4_L5':[],    

                'Right_Subarticular_Stenosis_L5_S1':[],



                'Left_Neural_Foraminal_Narrowing_L1_L2':[],

                'Left_Neural_Foraminal_Narrowing_L2_L3':[],

                'Left_Neural_Foraminal_Narrowing_L3_L4':[],

                'Left_Neural_Foraminal_Narrowing_L4_L5':[],    

                'Left_Neural_Foraminal_Narrowing_L5_S1':[],

                'Right_Neural_Foraminal_Narrowing_L1_L2':[],

                'Right_Neural_Foraminal_Narrowing_L2_L3':[],

                'Right_Neural_Foraminal_Narrowing_L3_L4':[],

                'Right_Neural_Foraminal_Narrowing_L4_L5':[],    

                'Right_Neural_Foraminal_Narrowing_L5_S1':[],



                'Spinal_Canal_Stenosis_L1_L2':[],

                'Spinal_Canal_Stenosis_L2_L3':[],

                'Spinal_Canal_Stenosis_L3_L4':[],

                'Spinal_Canal_Stenosis_L4_L5':[],    

                'Spinal_Canal_Stenosis_L5_S1':[],
        
            }
            
        








            
        for unique_study_id, series_group in df.groupby('study_id'):

            unique_series_id_for_this_study_id_list = series_group['series_id'].unique().tolist()
            
            # print(unique_study_id)

            # print(unique_series_id_for_this_study_id_list)

            for each_series_id in unique_series_id_for_this_study_id_list:

                list_prediction_for_this_study_id_series_id=result[f'si_{unique_study_id}_sei_{each_series_id}']['prediction']

                for each_prediction in list_prediction_for_this_study_id_series_id:

                    predicted_condition=each_prediction['condition']

                    predicted_score=each_prediction['score_prediction']


                    study_ids_report[f'{unique_study_id}'][predicted_condition].append(predicted_score)









                # print(a)

                # print(b)

        





        #### beacasue for each condition we have number of report do the mean 


        for study_id in list(study_ids_report.keys()):

            all_conditions=list(study_ids_report[study_id].keys())

            for condition in all_conditions:

                all_score_predicted_for_this_condition=study_ids_report[study_id][condition]

                if len(all_score_predicted_for_this_condition)==0:



                    study_ids_report[study_id][condition]=[0,0,0]    

                else:


                    averages_list = [sum(x) / len(x) for x in zip(*all_score_predicted_for_this_condition)]


                    study_ids_report[study_id][condition]=averages_list

        
        # print(study_ids_report)


        csv_data = []

        for patient_id, conditions in study_ids_report.items():
            for condition, values in conditions.items():
                # Generate the row ID
                row_id = f"{patient_id}_{condition.lower()}"
                # Create a row for CSV
                csv_data.append([row_id] + values)

        # Write to CSV
        csv_filename = "./output.csv"
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["row_id", "normal_mild", "moderate", "severe"])
            # Write the data
            writer.writerows(csv_data)

        print(f"Data has been written to {csv_filename}")





            














predictior(test_csv='../Data/test_series_descriptions.csv',test_images='../Data/test_images',model_path='./trained_model')


















