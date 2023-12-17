import pandas as pd
import numpy as np

diabetesPath = '/Users/araltkk/Documents/Bilkent/2023-2024 Fall/EEE 485 - Statistical Learning and Data Analytics/Project/Code Folder/Datasets/diabetes.csv'
meanDiabetesPath = '/Users/araltkk/Documents/Bilkent/2023-2024 Fall/EEE 485 - Statistical Learning and Data Analytics/Project/Code Folder/Datasets/mean_valued_diabetes.csv'
zeroDiabetesPath = '/Users/araltkk/Documents/Bilkent/2023-2024 Fall/EEE 485 - Statistical Learning and Data Analytics/Project/Code Folder/Datasets/zero_removed_diabetes.csv'

class DiabetesDataProcessor:
    def __init__(self, diabetes_file= diabetesPath, mean_diabetes_file= meanDiabetesPath, zero_removed_diabetes_file= zeroDiabetesPath):
        # Import the datasets.
        self.diabetesDataFrame = pd.read_csv(diabetes_file)
        self.meanDiabetesDataFrame = pd.read_csv(mean_diabetes_file)
        self.removedDiabetesDataFrame = pd.read_csv(zero_removed_diabetes_file)

    # Execute integer labeling on the features of the dataset.
    def integer_labeling(self, dataframe):
        return dataframe.replace(['Pregnancies', 'Glucose', 'Blood Pressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
                                 [0, 1, 2, 3, 4, 5, 6, 7, 8]).values

    # Create test and training datasets using a test fraction and the original dataset.
    def create_test_and_training_datasets(self, dataset, test_fraction=0.3):
        dataset_size = int(len(dataset) * test_fraction)
        training_data = dataset[:-dataset_size, :]
        test_data = dataset[-dataset_size:, :]
        return training_data, test_data
    
    # Apply Standard scaling.
    def standardScaling(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_scaled = (X - mean) / std
        
        
        return X_scaled

    def process_datasets(self):
        # Execute integer labeling on features and do random shuffling on datasets.
        shuffled_Diabetes = self.integer_labeling(self.diabetesDataFrame)
        shuffled_Mean_Diabetes = self.integer_labeling(self.meanDiabetesDataFrame)
        shuffled_Removed_Diabetes = self.integer_labeling(self.removedDiabetesDataFrame)

        # Create the training datasets.
        diabetesTraining, diabetesTest = self.create_test_and_training_datasets(shuffled_Diabetes)
        meanDiabetesTraining, meanDiabetesTest = self.create_test_and_training_datasets(shuffled_Mean_Diabetes)
        removedDiabetesTraining, removedDiabetesTest = self.create_test_and_training_datasets(shuffled_Removed_Diabetes)

        # Choose the features and labels from the original diabetes dataset.
        Features = diabetesTraining[:, :-1]
        Features = self.standardScaling(Features)
        Labels = diabetesTraining[:, -1]

        # Choose the features and labels from the mean-valued diabetes dataset.
        meanFeatures = meanDiabetesTraining[:, :-1]
        meanFeatures = self.standardScaling((meanFeatures))
        meanLabels = meanDiabetesTraining[:, -1]

        # Choose the features and labels from the zero-removed diabetes dataset.
        zeroFeatures = removedDiabetesTraining[:, :-1]
        zeroFeatures = self.standardScaling((zeroFeatures))
        zeroLabels = removedDiabetesTraining[:, -1]

        # Choose the test features and labels from the original dataset.
        test_Features = diabetesTest[:, :-1]
        test_Features = self.standardScaling((test_Features))
        test_Labels = diabetesTest[:, -1]

        # Choose the test features and labels from the mean-valued dataset.
        mean_test_Features = meanDiabetesTest[:, :-1]
        mean_test_Features = self.standardScaling(mean_test_Features)
        mean_test_Labels = meanDiabetesTest[:, -1]

        # Choose the test features and labels from the zero-removed dataset.
        zero_test_Features = removedDiabetesTest[:, :-1]
        zero_test_Features = self.standardScaling(zero_test_Features)
        zero_test_Labels = removedDiabetesTest[:, -1]

        return (Features, Labels, meanFeatures, meanLabels, 
                zeroFeatures, zeroLabels, diabetesTest, meanDiabetesTest, removedDiabetesTest,
                test_Features, test_Labels, mean_test_Features, mean_test_Labels,
                zero_test_Features, zero_test_Labels)


