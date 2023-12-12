# Medications response
 Utilizing Decsion tree to show the medication response on patients
Develop a classification model using Decision Tree Algorithm
I use this classification algorithm to build a model from the historical data of patients, and their response to different medications. Then I use the trained decision tree to predict the class of an unknown patient, or to find a proper drug for a new patient.
About the dataset¶
Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug c, Drug x and y.

Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness. The features of this dataset are Age, Sex, Blood Pressure, and the Cholesterol of the patients, and the target is the drug that each patient responded to.

It is a sample of multiclass classifier, and you can use the training part of the dataset to build a decision tree, and then use it to predict the class of an unknown patient, or to prescribe a drug to a new patient.

Downloading the Data
To download the data, you can use !wget to download it from IBM Object Storage.
path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'