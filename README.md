# eye_disorders
Dataset Link: https://drive.google.com/drive/folders/1-14F4q1D8pSUH1OVzlUk7umA0FH9No6a?usp=drive_link

1. **Import Required Libraries:**
   - Import necessary libraries such as NumPy, pandas, OpenCV, matplotlib, and scikit-learn modules.

2. **Define File Paths:**
   - Specify the paths for the training and testing datasets.

3. **Load and Organize Dataset:**
   - Read the list of files in the training and testing directories.
   - Create empty lists to store image paths and corresponding labels.

4. **Generate Image Paths and Labels:**
   - Iterate through each class folder in the training and testing datasets.
   - Build the full path for each image file and store it along with the corresponding label.

5. **Create DataFrames:**
   - Create two pandas DataFrames, `train_df` and `test_df`, to store image paths and labels.

6. **Map Class Labels:**
   - Map class labels to numerical values using a predefined dictionary (`class_mapping`).

7. **Save DataFrames as CSV:**
   - Save the generated DataFrames to CSV files (`training_data.csv` and `testing_data.csv`) for future use.

8. **Display Dataset Information:**
   - Display information about the training and testing DataFrames, such as column names, data types, and non-null counts.

9. **Random Seed Initialization:**
   - Set a random seed for reproducibility.

10. **Additional Information:**
    - Various commented-out sections and import statements are included in the code, which are either unused or optional based on the user's needs.

11. **File Paths Output:**
    - The paths for training and testing datasets are specified and hardcoded; users should modify these paths based on their own directory structure.

12. **End of Code:**
    - The code concludes with a commented-out line that saves predictions to a NumPy file, which appears to be unused in the provided snippet.
   
The training code: 
### Steps to Describe the Code :

1. **Author Information:**
   - The script starts with author information, specifying the author as "nour."

2. **Import Required Libraries:**
   - Import necessary libraries such as NumPy, pandas, OpenCV, matplotlib, TensorFlow, scikit-learn, and others.

3. **GPU Configuration:**
   - Check and configure GPU availability and memory settings.
   - Optional GPU memory fraction settings are included for customization.

4. **Data Loading:**
   - Load training and testing datasets using pandas DataFrames from CSV files.

5. **Biomedical Image Loading and Preprocessing:**
   - Load biomedical images, resize them, and normalize pixel values.
   - The script handles errors in reading and processing images and prints the number of skipped images.

6. **Model Architecture:**
   - Define and compile three models (ResNet101, ResNet50, and VGG16) for image classification.
   - Transfer learning is used by freezing pre-trained layers and adding custom classifiers.

7. **Model Training:**
   - Train each model using the training dataset and validate on a validation set.
   - Save the trained models to HDF5 files.

8. **Model Evaluation:**
   - Evaluate the models on the validation set using accuracy, F1-score, precision, recall, confusion matrix, and a classification report.
   - Save evaluation results to text files.

9. **Visualize Results:**
   - Create bar charts displaying validation accuracy and F1-score for each model.

10. **Test Set Evaluation:**
    - Apply the trained models to the test set.
    - Evaluate each model's performance on the test set using the same metrics as for the validation set.

11. **Evaluation Function:**
    - Define a function (`evaluate_model`) to calculate and print various evaluation metrics.

12. **Results Saving:**
    - Save the evaluation results to text files for documentation and future reference.

13. **Additional Information:**
    - The script contains informative comments, including instructions for replacing actual paths and labels.

14. **Visualization:**
    - Plot bar charts displaying validation accuracy and F1-score for each model.

15. **End of Code:**
    - The script concludes with the completion of model training, evaluation, and visualization.



