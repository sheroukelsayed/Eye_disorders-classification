#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 20:42:26 2022

@author: nour
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 22:20:21 2022

@author: nour
"""
import numpy as np
import pandas as pd
import io
import os
import cv2

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn.metrics import confusion_matrix,f1_score,classification_report
import matplotlib.pyplot as plt

import itertools

from sklearn.utils import shuffle


#from tensorflow.keras.models import Model

import codecs
import os
#from tensorflow.keras import backend as K
#from keras_bert import load_trained_model_from_checkpoint
import numpy as np
import pandas as pd

#import datasets

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from sklearn.metrics import accuracy_score, f1_score


import random
from ultralytics import YOLO

random.seed(10)
# Additional libraries for specific models
#from yolov5 import YOLOv5
#from transformers import  ViTForImageClassification
from tensorflow.keras.applications import InceptionV3, DenseNet121
#from vit_keras import vit






from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from tensorflow.keras import backend as K
import tensorflow as tf

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Set GPU memory fraction (optional)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # If GPUs are available, set memory growth for each GPU
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# If you want to limit GPU memory fraction, you can use the following code
# This is optional and depends on your specific requirements
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
# )

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Assuming you have already defined and compiled your model

# Define the EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,           # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
    verbose=1             # Set verbose to 1 to see log messages
)


train='/home1/nour/sherouk/New_task/DATASET_101/Train/'
test='/home1/nour/sherouk/New_task/DATASET_101/Test/'
train_df = pd.read_csv('/home1/nour/sherouk/New_task/DATASET_101/training_data.csv')

# Shuffle the DataFrame
train_df_shuffled = shuffle(train_df, random_state=42)  # You can set a specific random_state for reproducibility

train_df_shuffled.head(5)

test_df = pd.read_csv('/home1/nour/sherouk/New_task/DATASET_101/testing_data.csv')



X=[]
Y=[]

for index, row in train_df_shuffled.iterrows():

    image = cv2.imread(row['image_path'])


    # Check if the image is None (indicating failure to read)
    if image is None:
        print(f"Error reading image: {row['image_path']}")
        break;
    else:
    # Resize the image
      image = cv2.resize(image, (224, 224))
     
      label = row['label']
      X.append(image)
      Y.append(label)
'''
X_test=[]
Y_test=[]
for index, row in test_df.iterrows():

    image = cv2.imread(row['image_path'])


    # Check if the image is None (indicating failure to read)
    if image is None:
        print(f"Error reading image: {row['image_path']}")
        break;
    else:
    # Resize the image
      image = cv2.resize(image, (224, 224))
      
      label = row['label']
      X_test.append(image)
      Y_test.append(label)
'''
# Convert lists to numpy arrays
X = np.array(X)
y = np.array(Y)
'''
X_test=np.array(X_test)
Y_test=np.array(Y_test)
'''
# Convert labels to numerical format
y_categorical =to_categorical(train_df_shuffled['label'])


# Split the dataset into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)
'''
X_test=[]
Y_test=[]
# Assuming you have X_valid and y_valid as your validation data
validation_data = (X_valid, y_valid)
for index, row in test_df.iterrows():

    image = cv2.imread(row['image_path'])


    # Check if the image is None (indicating failure to read)
    if image is None:
        print(f"Error reading image: {row['image_path']}")
        break;
    else:
    # Resize the image
      image = cv2.resize(image, (224, 224))
      print(image.shape)
      label = row['label']
      X_test.append(image)
      Y_test.append(label)


'''
# Function to evaluate model performance
def evaluate_model(predictions, true_labels, model_name):
    f1 = f1_score(true_labels, predictions, average='weighted')
    accuracy = accuracy_score(true_labels, predictions)
    recall = tf.keras.metrics.Recall()(true_labels, predictions)
    precision = tf.keras.metrics.Precision()(true_labels, predictions)

    cm = confusion_matrix(true_labels, predictions)
    cr = classification_report(true_labels, predictions)

    # Save results to a text file
    with open(f"/home1/nour/sherouk/New_task/DATASET_101/{model_name}_evaluation_results.txt", "w") as file:
        file.write(f"F1 Score: {f1}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"Precision: {precision}\n")
        file.write("Confusion Matrix:\n")
        file.write(str(cm))
        file.write("\n\nClassification Report:\n")
        file.write(cr)

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Load your dataset and shuffle
# ...

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(Y)

# Convert labels to numerical format
y_categorical = to_categorical(train_df_shuffled['label'])

# Split the dataset into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Assuming you have X_valid and y_valid as your validation data
validation_data = (X_valid, y_valid)

# Loop through each model
'''
# YOLOv5
yolo_model = YOLO('yolov8n-cls.pt') # load a pretrained model (recommended for training)
# Load a model

for layer in yolo_model.layers:
    layer.trainable = False
x = GlobalAveragePooling2D()(yolo_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(6, activation='softmax')(x)
yolo_freeze = Model(inputs=yolo_model.input, outputs=output)
yolo_freeze.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
yolo_freeze.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=validation_data)
yolo_freeze.save('/home1/nour/sherouk/New_task/DATASET_101/yolov5.h5')
yolo_val_preds = yolo_freeze.predict(X_valid)
evaluate_model(np.argmax(yolo_val_preds, axis=1), np.argmax(y_valid, axis=1), "yolov5")
# Inception
inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in inception_model.layers:
    layer.trainable = False
x = GlobalAveragePooling2D()(inception_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(6, activation='softmax')(x)
inception_freeze = Model(inputs=inception_model.input, outputs=output)
inception_freeze.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
inception_freeze.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=validation_data)
inception_freeze.save('/home1/nour/sherouk/New_task/DATASET_101/inception.h5')
inception_val_preds = inception_freeze.predict(X_valid)
evaluate_model(np.argmax(inception_val_preds, axis=1), np.argmax(y_valid, axis=1), "inception")
'''
# DenseNet
densenet_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in densenet_model.layers:
    layer.trainable = False
x = GlobalAveragePooling2D()(densenet_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(6, activation='softmax')(x)
densenet_freeze = Model(inputs=densenet_model.input, outputs=output)
densenet_freeze.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
densenet_freeze.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=validation_data)
densenet_freeze.save('/home1/nour/sherouk/New_task/DATASET_101/densenet.h5')
densenet_val_preds = densenet_freeze.predict(X_valid)
evaluate_model(np.argmax(densenet_val_preds, axis=1), np.argmax(y_valid, axis=1), "densenet")






