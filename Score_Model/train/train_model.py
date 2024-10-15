
'''

Train model for severity classification


'''
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input, GlobalAveragePooling2D,BatchNormalization,add,Activation,SeparableConv2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np
import os








class training:
    
    def __init__(self,



        condition='',

        fold='',

        learning_rate=0.001,

        epochs=100,
    

        batch=8,

        patients=20,



    
    
    ):



        self.condition=condition

        self.fold=fold



        self.epochs=epochs

        self.batch=batch,

        self.patients=patients,

        

        self.create_folder()

        self.read_csv()


        ### Load images

        train_subjects,train_labels,val_subjects,val_labels=self.read_csv()

        
        X_train, y_train = self.load_image(train_subjects, train_labels, f"../data_prepration/{self.condition}/fold_{self.fold}/train")

        X_val, y_val = self.load_image(val_subjects, val_labels, f"../data_prepration/{self.condition}/fold_{self.fold}/val")


        model=self.model_architecture((32,32))


        print(model.summary())



        class_weights,alpha=self.calculate_class_weights(y_train)
        alpha=list(alpha)
        class_weights=list(class_weights)
        

        
        y_train=self.one_hot_labels(y_train)
        y_val=self.one_hot_labels(y_val)


        sgd_optimizer=SGD(
            learning_rate=learning_rate,
            momentum=0.9)
        
        self.compile_fit_model(model,X_train,y_train,X_val,y_val,alpha,class_weights,sgd_optimizer)


        self.evaluate_the_model(model,X_val,y_val)










    def create_folder(self):

        self.fold_path = Path(f'./results/{self.condition}/fold_{self.fold}')

        if not self.fold_path.exists():
            self.fold_path.mkdir(parents=True, exist_ok=True)








    def read_csv(self):

        train_data = pd.read_csv(f'../data_prepration/{self.condition}_label/fold_{self.fold}/{self.condition}_augmented_labels.csv')
        val_data = pd.read_csv(f'../data_prepration/{self.condition}_label/fold_{self.fold}/{self.condition}_val_labels.csv')


        train_subjects = train_data['subject']
        train_labels = train_data['label']

        val_subjects = val_data['subject']
        val_labels = val_data['label']


        return train_subjects,train_labels,val_subjects,val_labels






    def load_image(self,subjects, labels, directory):
        img_list = []
        for subject in subjects:
            img_path = os.path.join(directory, subject )

            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")

            try:
                img = load_img(img_path, target_size=(32, 32))  # Resize to 32x32
                img_array = img_to_array(img)
                img_list.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

        



        return np.array(img_list), np.array(labels)


    
    def one_hot_labels(self,labels):

                 
        return to_categorical(labels - 1, num_classes=3)









    def model_architecture(self,image_size):
      
        
        # Input layer with specified image size and a single channel
        input_layer = Input(shape=image_size + (3,))
        
        # Initial convolution layer
        initial_conv = input_layer

        # First Convolution Block
        x = Conv2D(32, 3, strides=1, padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        conv_2 = x  # Save the output for residual connections
        
        previous_block_output = x  # Initialize the previous block output for residual connections
        
        # Downsampling Blocks
        for num_filters in [64, 128, 256]:
            # Convolution layers
            x = Activation("relu")(x)
            x = Conv2D(num_filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            
            x = Activation("relu")(x)
            x = Conv2D(num_filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            
            # Additional layers for higher number of filters
            if num_filters in [128, 256]:
                x = Activation("relu")(x)
                x = Conv2D(num_filters, 3, padding="same")(x)
                x = BatchNormalization()(x)
                
                x = Activation("relu")(x)
                x = Conv2D(num_filters, 3, padding="same")(x)
                x = BatchNormalization()(x)
            
            # Downsampling with max pooling
            x = MaxPooling2D(3, strides=2, padding="same")(x)
            
            # Residual connection
            residual = Conv2D(num_filters, 1, strides=2, padding="same")(previous_block_output)
            x = add([x, residual])  
            
            # Save outputs for upsampling residuals
            if num_filters == 64:
                conv_3 = x
            elif num_filters == 128:
                conv_4 = x
            
            previous_block_output = x  # Update previous block output

        # Global Average Pooling layer
        x = GlobalAveragePooling2D()(x)

        # Dense layer for classification
        x = Dense(256, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        # x = Dropout(0.5)(x)  # Dropout for regularization

        # Output layer for classification (adjust num_classes based on your problem)
        output_layer = Dense(3, activation="softmax")(x)

        # Construct the model
        model = Model(inputs=input_layer, outputs=output_layer)


        return model








    def calculate_class_weights(self,y):
        """
        Calculate class weights based on the frequency of each class.
        
        Parameters:

        - y: Array-like, shape (n_samples,) containing class labels.
        
        Returns:
        - A dictionary mapping class labels to weights.
        """
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        class_weights = {c: total_samples / (len(classes) * count) for c, count in zip(classes, counts)}



        ### Alpha
        # class_counts = np.bincount(y)  # For binary or multi-class
        # total_count = len(y)
        # class_frequencies = class_counts / total_count

        # alpha = 1.0 / class_frequencies


        # alpha = alpha / np.sum(alpha) 

        print(list(counts))
        class_frequencies=[count /total_samples for count in list(counts)]

        alpha=[1.0 - class_frequence for class_frequence in class_frequencies]
        alpha= alpha / np.sum(alpha)


        return class_weights ,alpha




    def focal_loss_with_class_weights(self,gamma=2., alpha=None, class_weights=None):
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







    def compile_fit_model(self,model,X_train,y_train,X_val,y_val,alpha,class_weights,opt):

        model.compile(optimizer=opt, loss=self.focal_loss_with_class_weights(gamma=3.0, alpha=alpha, class_weights=class_weights), metrics=['accuracy'])



        checkpoint = ModelCheckpoint(filepath=f'./results/{self.condition}/fold_{self.fold}/best.keras', 
                             monitor='val_loss', 
                             save_best_only=True, 
                             mode='min', 
                             verbose=1)

# 2. EarlyStopping: Stop training if validation loss doesn't improve for 'patience' epochs
        early_stopping = EarlyStopping(monitor='val_loss', 
                                    patience=20,  # Number of epochs to wait for improvement
                                    mode='min', 
                                    verbose=1)



        x = model.fit(X_train, y_train, batch_size=8, epochs=self.epochs, verbose=1,
                        validation_data=(X_val,y_val),shuffle=True,callbacks=[checkpoint, early_stopping])  


    







    def evaluate_the_model(self,model,X_val,y_val):


        model.load_weights(f'./results/{self.condition}/fold_{self.fold}/best.keras')

        predictions = model.predict(X_val, batch_size=1)
        predicted_classes = np.argmax(predictions, axis=1)

        # Save predictions and true labels to a CSV file for each subject
        results_df = pd.DataFrame({
            'Subject': np.arange(len(X_val)),  # Assuming subject index corresponds to rows in X_val
            'True Label': np.argmax(y_val, axis=1),  # Assuming y_val is one-hot encoded
            'Predicted Label': predicted_classes
        })
        results_df.to_csv(f'./results/{self.condition}/fold_{self.fold}/validation_predictions.csv', index=False)

        # Calculate classification metrics
        accuracy = accuracy_score(np.argmax(y_val, axis=1), predicted_classes)
        precision = precision_score(np.argmax(y_val, axis=1), predicted_classes, average='weighted')
        recall = recall_score(np.argmax(y_val, axis=1), predicted_classes, average='weighted')
        f1 = f1_score(np.argmax(y_val, axis=1), predicted_classes, average='weighted')

        # Save metrics to a text file
        with open(f'./results/{self.condition}/fold_{self.fold}/validation_metrics.txt', 'w') as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1}\n")

        print("Validation predictions saved to 'validation_predictions.csv'")
        print("Classification metrics saved to 'validation_metrics.txt'")

                        













import argparse

parser = argparse.ArgumentParser(description='Model Training Parameters')  


parser.add_argument('--condition', type=str, help='condition')

parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')

parser.add_argument('--fold', type=int, default=0, help='the fold that you want to train')

parser.add_argument('--patients', type=int, default=20, help='Number of patients')

parser.add_argument('--batch', type=int, default=8, help='Batch size')

args=parser.parse_args()










training(
        condition=args.condition,

        fold=args.fold,

        learning_rate=0.001,

        epochs=args.epochs,

        batch=args.batch,

        patients=args.patients,



    
    
    )





# training(
#         condition='Subarticular_Stenosis',

#         fold=0,

#         learning_rate=0.001,

#         epochs=100,

#         batch=8,

#         patients=20,



    
    
#     )

# python3 train_model.py  --condition Neural_Foraminal_Narrowing --epochs 200  --fold 0 --patients 20 --batch 8






# cd /users/amousavi/Challenge/Lumbar-Spine-Degenerative-Classification//Score_Model/train/

# ml compiler/python/3.11.2
