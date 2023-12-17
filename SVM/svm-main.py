# Import the libraries and classes that will be used. 
from dataProcessing import DiabetesDataProcessor # Hand written data processing class. 
from supportVectorMachine import SupportVectorMachine # Hand written Support Vector Machine Classifier.
from performanceMetrics import PerformanceMetrics # Hand written Performance Metrics class.
from sklearn.svm import SVC # Pre-defined library SVM Classifier for comparison. 

# Function to train, predict and evaluate SVM models
def evaluate_svm_models(original_features, original_labels, test_features, test_labels, model_description):
    # Initiate the models
    custom_svm = SupportVectorMachine()
    sklearn_svm = SVC()

    # Train the custom SVM classifier
    custom_svm.fit(original_features, original_labels)

    # Fit the scikit-learn SVM classifier
    sklearn_svm.fit(original_features, original_labels)

    # Make predictions with both classifiers
    custom_predictions = custom_svm.predict(test_features)
    sklearn_predictions = sklearn_svm.predict(test_features)

    # Evaluate the performance
    custom_metrics = PerformanceMetrics(test_labels, custom_predictions)
    sklearn_metrics = PerformanceMetrics(test_labels, sklearn_predictions)

    # Print the performance metrics
    print(f"\nHand-written SVM Performance on the {model_description} dataset:")
    print("True Positive Rate:", round(custom_metrics.truePositiveRate(), 3))
    print("False Positive Rate:", round(custom_metrics.falsePositiveRate(), 3))
    print("F1 Score:", round(custom_metrics.f1Score(), 3))
    print("Accuracy:", round(custom_metrics.Accuracy(), 3))
    print("Confusion Matrix:")
    print(custom_metrics.ConfusionMatrix())

    print(f"\nScikit-learn SVM Performance on the {model_description} dataset:")
    print("True Positive Rate:", round(sklearn_metrics.truePositiveRate(), 3))
    print("False Positive Rate:", round(sklearn_metrics.falsePositiveRate(), 3))
    print("F1 Score:", round(sklearn_metrics.f1Score(), 3))
    print("Accuracy:", round(sklearn_metrics.Accuracy(), 3))
    print("Confusion Matrix:")
    print(sklearn_metrics.ConfusionMatrix())

# Create an instance of the data processing class
data_processor = DiabetesDataProcessor()
(Features, Labels, meanFeatures, meanLabels, 
 zeroFeatures, zeroLabels, _, _, _, 
 test_Features, test_Labels, mean_test_Features, mean_test_Labels, 
 zero_test_Features, zero_test_Labels) = data_processor.process_datasets()

# Evaluate on original dataset
evaluate_svm_models(Features, Labels, test_Features, test_Labels, "original")

# Evaluate on mean-value replaced dataset
evaluate_svm_models(meanFeatures, meanLabels, mean_test_Features, mean_test_Labels, "mean-value replaced")

# Evaluate on zero-row removed dataset
evaluate_svm_models(zeroFeatures, zeroLabels, zero_test_Features, zero_test_Labels, "zero-row removed")




