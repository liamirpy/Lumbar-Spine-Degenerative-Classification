
'''

Train VGG model for severity classification


'''





from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input, GlobalAveragePooling2D,BatchNormalization,add,Activation,SeparableConv2D
from tensorflow.keras.optimizers import Adam,AdamW,schedules,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K


import numpy as np
import pandas as pd
import os



train_data = pd.read_csv('../data_prepration/Subarticular_Stenosis_label/fold_2/Subarticular_Stenosis_augmented_labels.csv')
val_data = pd.read_csv('../data_prepration/Subarticular_Stenosis_label/fold_2/Subarticular_Stenosis_val_labels.csv')

train_subjects = train_data['subject']
train_labels = train_data['label']

val_subjects = val_data['subject']
val_labels = val_data['label']






def load_image(subjects, labels, directory):
    img_list = []
    for subject in subjects:
        img_path = os.path.join(directory, subject )

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue 

        try:
            img = load_img(img_path, target_size=(32, 32))  # Resize to 224x224
            img_array = img_to_array(img)
            # print(f"Loaded image shape for {subject}: {img_array.shape}")
            img_list.append(img_array)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")



    #     img = load_img(img_path)  # Resize to 224x224 as VGG expects this size
    #     img_array = img_to_array(img)
    #     img_list.append(img_array)

    return np.array(img_list), np.array(labels)





print(train_labels)
# Load images for training and validatio
X_train, y_train = load_image(train_subjects, train_labels, "../data_prepration/Subarticular_Stenosis/fold_2/train")
X_val, y_val = load_image(val_subjects, val_labels, "../data_prepration/Subarticular_Stenosis/fold_2/val")


# y_train=np.array(train_labels)

# Convert labels to categorical (one-hot encoding)
y_train_one_hot = to_categorical(y_train - 1, num_classes=3)
y_val = to_categorical(y_val - 1, num_classes=3)





def build_model(image_size):
    """
    Build the RAX_NET model.
    
    Returns:
    - model: Keras Model, the constructed neural network model
    """
    
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
        x = SeparableConv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        
        # Additional layers for higher number of filters
        if num_filters in [128, 256]:
            x = Activation("relu")(x)
            x = Conv2D(num_filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            
            x = Activation("relu")(x)
            x = SeparableConv2D(num_filters, 3, padding="same")(x)
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
    x = Dropout(0.5)(x)  # Dropout for regularization

    # Output layer for classification (adjust num_classes based on your problem)
    output_layer = Dense(3, activation="softmax")(x)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)


    return model


# import numpy as np

def build_model_2(image_size):
    """
    Build the RAX_NET model.
    
    Returns:
    - model: Keras Model, the constructed neural network model
    """
    
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

model = build_model_2((32,32))

print(model.summary())







initial_learning_rate=10e-3
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9
)

# Define AdamW optimizer
opt = AdamW(
    learning_rate=initial_learning_rate,
  
)






import numpy as np

def calculate_class_weights(y):
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



import tensorflow as tf
from tensorflow.keras import backend as K

def focal_loss_with_class_weights(gamma=2., alpha=None, class_weights=None):
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








def focal_loss_with_class_weights_2(gamma=2., alpha=0.25, class_weights=None):
    """
    Focal Loss function for classification tasks with class weights.
    
    Args:
    - gamma: focusing parameter for modulating factor (1-p)
    - alpha: balancing factor for classes
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
        
        # If class weights are provided, multiply them with the cross entropy loss
        if class_weights is not None:
            class_weights_tensor = K.constant(class_weights, dtype=tf.float32)
            # Apply class weights to the loss
            weights = y_true * class_weights_tensor
            cross_entropy_loss *= weights

        # Apply alpha to the loss for each class
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Final focal loss computation
        loss = alpha_factor * modulating_factor * cross_entropy_loss
        return K.sum(loss, axis=-1)

    return focal_loss




def focal_loss_with_class_weights_3(gamma=2., alpha=None, class_weights=None):
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
        # if class_weights is not None:
        #     class_weights_tensor = K.constant(class_weights, dtype=tf.float32)
        #     weights = y_true * class_weights_tensor
        #     cross_entropy_loss *= weights

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








class_weights,alpha=calculate_class_weights(y_train)
alpha=list(alpha)
class_weights=list(class_weights)
print(alpha)



# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 81.26




# model.compile(optimizer=opt, loss=focal_loss_with_class_weights(gamma=2.0, alpha=alpha, class_weights=class_weights), metrics=['accuracy'])

#### NOT GOOD



# model.compile(optimizer='adam', loss=focal_loss_with_class_weights(gamma=2.0, alpha=alpha, class_weights=class_weights), metrics=['accuracy'])


### Its fine near to 80 but has flactuation



# model.compile(optimizer='SGD', loss=focal_loss_with_class_weights(gamma=2.0, alpha=alpha, class_weights=class_weights), metrics=['accuracy'])



### Tend to over fitting and reach 80 


s=SGD(
    learning_rate=0.001,
    momentum=0.9)

# model.compile(optimizer=s, loss=focal_loss_with_class_weights(gamma=2.0, alpha=alpha, class_weights=class_weights), metrics=['accuracy'])


### Tend to over fitting and reach 80 

# model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])


### 81.24





###################### DECISION ABOUT LOSS FUNCTION 


'''
1- the  adamw is not good. 

2- 

###### SGD IS BEST ##### 


'''






# model.compile(optimizer='SGD', loss=focal_loss_with_class_weights(gamma=2.0, alpha=0.25, class_weights=class_weights), metrics=['accuracy'])




# model.compile(optimizer='SGD', loss=focal_loss_with_class_weights_2(gamma=2.0, alpha=0.25, class_weights=class_weights), metrics=['accuracy'])


model.compile(optimizer=s, loss=focal_loss_with_class_weights_3(gamma=3.0, alpha=alpha, class_weights=class_weights), metrics=['accuracy'])




import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))







x = model.fit(X_train, y_train_one_hot, batch_size=8, epochs=300, verbose=1,
                validation_data=(X_val,y_val),shuffle=True
                
                
                
                
                
                
                )










# cd /users/amousavi/Challenge/Lumbar-Spine-Degenerative-Classification//Score_Model/train/

# ml compiler/python/3.11.2
