# Import the libraries and classes that will be used.
from performanceMetrics import PerformanceMetrics # Hand written performance metrics class.
from gaussianNaiveBayes import gaussianNaiveBayes # Hand written Gaussian Naive Bayes classifier class.
from dataProcessing import DiabetesDataProcessor # Hand written data processing class.
from sklearn.naive_bayes import GaussianNB # Pre-defined Gaussian Naive Bayes classifier for comparison.


# Function to train, predict and evaluate Gaussian Naive Bayes models. 
def evaluate_gnb_models(original_features, original_labels, test_features, test_labels, model_description):
    # Initiate the models
    custom_gnb = gaussianNaiveBayes()
    sklearn_gnb = GaussianNB()

    # Train the custom Gaussian Naive Bayes classifier. 
    custom_gnb.fitData(original_features, original_labels)

    # Make predictions with the custom classifier. 
    custom_predictions = custom_gnb.Predictor(test_features)

    # Evaluate the performance of the custom classifier. 
    custom_metrics = PerformanceMetrics(test_labels, custom_predictions)

    # Train scikit-learn's Gaussian Naive Bayes classifier. 
    sklearn_gnb.fit(original_features, original_labels)

    # Make predictions with scikit-learn's classifier. 
    sklearn_predictions = sklearn_gnb.predict(test_features)

    # Evaluate the performance of scikit-learn's classifier. 
    sklearn_metrics = PerformanceMetrics(test_labels, sklearn_predictions)

    # Print the performance metrics. 
    print(f"\nHand-written Gaussian Naive Bayes Performance on the {model_description} dataset:")
    print_performance(custom_metrics)

    print(f"\nScikit-learn Gaussian Naive Bayes Performance on the {model_description} dataset:")
    print_performance(sklearn_metrics)

# Function to print performance metrics. 
def print_performance(metrics):
    print("True Positive Rate:", round(metrics.truePositiveRate(), 3))
    print("False Positive Rate:", round(metrics.falsePositiveRate(), 3))
    print("F1 Score:", round(metrics.f1Score(), 3))
    print("Accuracy:", round(metrics.Accuracy(), 3))
    print("Confusion Matrix:")
    print(metrics.ConfusionMatrix())

# Create an instance of the data processing class. 
data_processor = DiabetesDataProcessor()
(Features, Labels, meanFeatures, meanLabels, 
 zeroFeatures, zeroLabels, _, _, _, 
 test_Features, test_Labels, mean_test_Features, mean_test_Labels, 
 zero_test_Features, zero_test_Labels) = data_processor.process_datasets()

# Evaluate on original dataset. 
evaluate_gnb_models(Features, Labels, test_Features, test_Labels, "original")

# Evaluate on mean-value replaced dataset. 
evaluate_gnb_models(meanFeatures, meanLabels, mean_test_Features, mean_test_Labels, "mean-value replaced")

# Evaluate on zero-row removed dataset. 
evaluate_gnb_models(zeroFeatures, zeroLabels, zero_test_Features, zero_test_Labels, "zero-row removed")