# Import the necessary libraries and classes.
from dataProcessing import DiabetesDataProcessor  # Handwritten data processing class. 
from performanceMetrics import PerformanceMetrics  # Handwritten Performance Metrics class.
from sklearn.ensemble import RandomForestClassifier  # Pre-defined library Random Forest Classifier for comparison.
from randomForest import randomForestClassifier  # Hand-written Random Forest Classifier.

# Create an instance of the data processing class
data_processor = DiabetesDataProcessor()
(Features, Labels, meanFeatures, meanLabels, 
 zeroFeatures, zeroLabels, diabetesTest, meanDiabetesTest, removedDiabetesTest,
 test_Features, test_Labels, mean_test_Features, mean_test_Labels,
 zero_test_Features, zero_test_Labels) = data_processor.process_datasets()

# Define a function to evaluate and compare models
def evalRandomForest(original_features, original_labels, test_features, test_labels, model_description):
    # Initiate the models
    custom_rf = randomForestClassifier()
    sklearn_rf = RandomForestClassifier()

    # Train the Custom Random Forest classifier
    custom_rf.fitData(original_features, original_labels)

    # Fit the sklearn Random Forest classifier
    sklearn_rf.fit(original_features, original_labels)

    # Make predictions with both classifiers
    custom_rf_predictions = custom_rf.Predictor(test_features)
    sklearn_rf_predictions = sklearn_rf.predict(test_features)

    # Evaluate the performance
    custom_rf_metrics = PerformanceMetrics(test_labels, custom_rf_predictions)
    sklearn_rf_metrics = PerformanceMetrics(test_labels, sklearn_rf_predictions)

    # Print the performance metrics
    print(f"\nHand-written Random Forest Performance on {model_description} dataset:")
    print("True Positive Rate:", round(custom_rf_metrics.truePositiveRate(), 3))
    print("False Positive Rate:", round(custom_rf_metrics.falsePositiveRate(), 3))
    print("F1 Score:", round(custom_rf_metrics.f1Score(), 3))
    print("Accuracy:", round(custom_rf_metrics.Accuracy(), 3))
    print("Confusion Matrix:")
    print(custom_rf_metrics.ConfusionMatrix())

    print(f"\nScikit-learn Random Forest Performance {model_description} dataset:")
    print("True Positive Rate:", round(sklearn_rf_metrics.truePositiveRate(), 3))
    print("False Positive Rate:", round(sklearn_rf_metrics.falsePositiveRate(), 3))
    print("F1 Score:", round(sklearn_rf_metrics.f1Score(), 3))
    print("Accuracy:", round(sklearn_rf_metrics.Accuracy(), 3))
    print("Confusion Matrix:")
    print(sklearn_rf_metrics.ConfusionMatrix())
    
# Evaluate on original dataset
evalRandomForest(Features, Labels, test_Features, test_Labels, 'original')

# Evaluate on mean-value replaced dataset
evalRandomForest(meanFeatures, meanLabels, mean_test_Features, mean_test_Labels, 'mean-value replaced')

# Evaluate on zero-row removed dataset
evalRandomForest(zeroFeatures, zeroLabels, zero_test_Features, zero_test_Labels, 'zero-row removed')





