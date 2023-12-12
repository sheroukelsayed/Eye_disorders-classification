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

import random


random.seed(10)




train='/home1/nour/sherouk/New_task/DATASET_101/Train/'
test='/home1/nour/sherouk/New_task/DATASET_101/Test/'
train_df = pd.DataFrame()
test_df=pd.DataFrame()
training_files=os.listdir(train)
test_files=os.listdir(test)

print(training_files)

image_paths = []
image_labels = []

for i in range(len(training_files)):
  classes= ["ACRIMA", "cataract", "Glaucoma", "ODIR-5K", "ORIGA", "retina_disease"]
  
  folder_path=train + training_files[i]

  files_folders=os.listdir(folder_path)

  for img in range(len(files_folders)):
    image_path=folder_path + '/'  +files_folders[img]
    image_label=classes[i]

    image_paths.append(image_path)
    image_labels.append(image_label)
 # Verify the lengths before assignment
print(len(image_paths))
print(len(image_labels))
train_df['image_path'] = image_paths
train_df['label'] = image_labels

train_df['image_path'][0]

# Map the 'label' column using the defined mapping
class_mapping = {"ACRIMA":0, "cataract":1, "Glaucoma":2, "ODIR-5K":3, "ORIGA":4, "retina_disease":5}

train_df['label'] = train_df['label'].map(class_mapping)

test_images_paths=[]
test_images_labels=[]

for i in range(len(test_files)):
  folder_path2=test+test_files[i]
  files_folders2=os.listdir(folder_path2)
  for img in range(len(files_folders2)):
    image_path2=folder_path2 + '/'  +files_folders2[img]
    image_label2=classes[i]

    test_images_paths.append(image_path2)
    test_images_labels.append(image_label2)
 # Verify the lengths before assignment
print(len(image_paths))
print(len(image_labels))
test_df['image_path'] = test_images_paths
test_df['label'] = test_images_labels

print(len(test_images_paths))
print(len(test_images_labels))



test_df['label'] = test_df['label'].map(class_mapping)

train_df.info()

test_df.info()

# Assuming train_df is your DataFrame
train_df.to_csv('/home1/nour/sherouk/New_task/DATASET_101/training_data.csv', index=False)
test_df.to_csv('/home1/nour/sherouk/New_task/DATASET_101/testing_data.csv', index=False)

test_df["image_path"][0]

#np.save(r"/home1/nour/sherouk/PHS_bert/predictions_prob_PHS_bert.npy", pred_prob)


   

 
