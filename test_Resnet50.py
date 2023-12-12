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
from keras.applications.resnet import preprocess_input



from sklearn.metrics import accuracy_score, f1_score


import random

from tensorflow.keras.models import load_model
random.seed(10)




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
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
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
'''
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
      image=image.astype('float32') / 255.0
      X.append(image)
      Y.append(label)

'''

X_test = []
Y_test = []
num_images_skipped = 0  # Counter for the number of images skipped due to errors
print("start load test data.....")
for index, row in test_df.iterrows():
    print(index)
    try:
        image = cv2.imread(row['image_path'])

        # Check if the image is None (indicating failure to read)
        if image is None:
            print(f"Error reading image: {row['image_path']}")
            num_images_skipped += 1
            continue

        # Resize the image
        image = cv2.resize(image, (224, 224))

        label = row['label']
        image=image.astype('float32') / 255.0
        X_test.append(image)
        Y_test.append(label)

    except Exception as e:
        print(f"Error processing image: {row['image_path']}")
        print(f"Error details: {str(e)}")
        num_images_skipped += 1
        continue

# Print the number of images skipped
print(f"Number of images skipped: {num_images_skipped}")
'''

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(Y)


# Convert labels to numerical format
y_categorical =to_categorical(train_df_shuffled['label'])


# Split the dataset into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)
'''
X_test=np.array(X_test)
Y_test=np.array(Y_test)
print(X_test.shape)
print(Y_test.shape)

#validation_data=(X_valid,y_valid )


'''
# Load the ResNet101 model pre-trained on ImageNet data
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add a custom classifier on top of the VGG16 model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(6, activation='softmax')(x)  # Assuming 2 classes (Stop and Non-Stop)

# Create the final model
Resnet101_freeze = Model(inputs=base_model.input, outputs=output)

# Compile the model
Resnet101_freeze.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
#Resnet101_freeze.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=validation_data )

#Resnet101_freeze.save('/home1/nour/sherouk/New_task/DATASET_101/resnet101.h5')
Resnet101_freeze= load_model('/home1/nour/sherouk/New_task/DATASET_101/resnet101.h5')
# Load the ResNet50 model pre-trained on ImageNet data
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add a custom classifier on top of the VGG16 model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(6, activation='softmax')(x)  # Assuming 2 classes (Stop and Non-Stop)

# Create the final model
Resnet50_freeze = Model(inputs=base_model.input, outputs=output)

# Compile the model
Resnet50_freeze.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
#Resnet50_freeze.fit(X_train, y_train, epochs=15, batch_size=32, validation_data= validation_data)
Resnet50_freeze=load_model('/home1/nour/sherouk/New_task/DATASET_101/resnet50.h5')
#Resnet50_freeze.save('/home1/nour/sherouk/New_task/DATASET_101/resnet50.h5')

#Load the VGG19 model pre-trained on ImageNet data
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add a custom classifier on top of the VGG19 model
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(6, activation='softmax')(x)  # Assuming 4 classes

# Create the final model
vgg19_freeze = Model(inputs=base_model.input, outputs=output)

# Compile the model
vgg19_freeze.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
#vgg19_freeze.fit(X_train, y_train, epochs=15, batch_size=32, validation_data= validation_data)
vgg19_freeze=load_model('/home1/nour/sherouk/New_task/DATASET_101/VGG19.h5')
#vgg19_freeze.save('/home1/nour/sherouk/New_task/DATASET_101/VGG19.h5')

# Evaluate on validation data
resnet50_val_preds = Resnet50_freeze.predict(X_valid)
resnet101_val_preds = Resnet101_freeze.predict(X_valid)
vgg19_val_preds = vgg19_freeze.predict(X_valid)

np.save('/home1/nour/sherouk/New_task/DATASET_101/resnet50_val_preds.npy', resnet50_val_preds)
np.save('/home1/nour/sherouk/New_task/DATASET_101/resnet101_val_preds.npy',resnet101_val_preds)
np.save('/home1/nour/sherouk/New_task/DATASET_101/vgg19_val_preds.npy', vgg19_val_preds)


resnet50_val_accuracy = accuracy_score(np.argmax(y_valid, axis=1), np.argmax(resnet50_val_preds, axis=1))
resnet101_val_accuracy = accuracy_score(np.argmax(y_valid, axis=1), np.argmax(resnet101_val_preds, axis=1))
vgg19_val_accuracy = accuracy_score(np.argmax(y_valid, axis=1), np.argmax(vgg19_val_preds, axis=1))

resnet50_val_f1= f1_score(np.argmax(y_valid, axis=1), np.argmax(resnet50_val_preds, axis=1), average='weighted')
resnet101_val_f1 = f1_score(np.argmax(y_valid, axis=1), np.argmax(resnet101_val_preds, axis=1), average='weighted')
vgg19_val_f1 = f1_score(np.argmax(y_valid, axis=1), np.argmax(vgg19_val_preds, axis=1), average='weighted')

# Repeat similar steps for test data

# Create a bar chart
labels = ['ResNet50', 'ResNet101', 'VGG16']
val_accuracies = [resnet50_val_accuracy, resnet101_val_accuracy, vgg19_val_accuracy]
val_f1_scores = [resnet50_val_f1, resnet101_val_f1, vgg19_val_f1]



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(labels, val_accuracies, color=['blue', 'green', 'orange'])
ax1.set_title('Validation Accuracy')

ax2.bar(labels, val_f1_scores, color=['blue', 'green', 'orange'])
ax2.set_title('Validation F1-Score')

plt.tight_layout()
plt.show()
'''
# Assuming X_test is your test data
batch_size = 32



from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(predictions, true_labels, Model_name):
    # Calculate F1 Score
    f1 = f1_score(true_labels, predictions, average='weighted')
    print("F1 Score:", f1)

    # Calculate Accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print("Accuracy:", accuracy)

    # Calculate Recall
    recall = recall_score(true_labels, predictions, average='weighted')
    print("Recall:", recall)

    # Calculate Precision
    precision = precision_score(true_labels, predictions, average='weighted')
    print("Precision:", precision)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    cr = classification_report(true_labels, predictions)
    print("Classification Report:")
    print(cr)

    # Save results to a text file
    with open("/home1/nour/sherouk/New_task/DATASET_101/"+ Model_name +"evaluation_results.txt", "w") as file:
        file.write(f"F1 Score: {f1}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"Precision: {precision}\n")
        file.write("Confusion Matrix:\n")
        file.write(str(cm))
        file.write("\n\nClassification Report:\n")
        file.write(cr)

# Assuming you have true labels for validation and test sets
# Replace y_valid and Y_test with your actual true labels
'''
# Evaluate on validation data
print("Evaluation on Validation Data:")
evaluate_model(np.argmax(resnet50_val_preds, axis=1), np.argmax(y_valid, axis=1),"resnet50")
evaluate_model(np.argmax(resnet101_val_preds, axis=1),np.argmax(y_valid, axis=1),"resnet101")
evaluate_model(np.argmax(vgg19_val_preds, axis=1),np.argmax(y_valid, axis=1),"vgg19")

'''

#save labels to numpy files
np.save('/home1/nour/sherouk/New_task/DATASET_101/vgg19_test_preds.npy', Y_test)
#load models
'''
print("predication of  Test Data(resnet50):")
print("start load model.....")
Resnet50_freeze= load_model('/home1/nour/sherouk/New_task/DATASET_101/resnet50.h5')
print("start predication process.....")
resnet50_test_preds = Resnet50_freeze.predict(X_test)
print("save predication file.....")
np.save('/home1/nour/sherouk/New_task/DATASET_101/resnet50_test_preds.npy', resnet50_test_preds)
print("Evaluation process and create text file of results.....")
evaluate_model(np.argmax(resnet50_test_preds, axis=1), Y_test,"resnet50_test")

print("predication of  Test Data(resnet101):")
print("start load model.....")
Resnet101_freeze= load_model('/home1/nour/sherouk/New_task/DATASET_101/resnet101.h5')
print("start predication process.....")
resnet101_test_preds = Resnet101_freeze.predict(X_test)
print("save predication file.....")
np.save('/home1/nour/sherouk/New_task/DATASET_101/resnet101_test_preds.npy',resnet101_test_preds)
print("Evaluation process and create text file of results.....")
evaluate_model(np.argmax(resnet101_test_preds, axis=1), Y_test,"resnet101_test")
'''

print("predication of  Test Data(resnet101):")
print("load model..........")
vgg19_freeze= load_model('/home1/nour/sherouk/New_task/DATASET_101/VGG19.h5')

print("start predication process.....")
vgg19_test_preds = vgg19_freeze.predict(X_test)
print("save predication file.....")
np.save('/home1/nour/sherouk/New_task/DATASET_101/vgg19_test_preds.npy', vgg19_test_preds)
print("Evaluation process and create text file of results.....")
evaluate_model(np.argmax(vgg19_test_preds, axis=1), Y_test,"vgg19_test")





 
