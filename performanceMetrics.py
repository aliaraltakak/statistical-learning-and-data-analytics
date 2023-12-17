# Import libraries. 
import numpy as np 

class PerformanceMetrics:

    # Initialize the required variables. 
    def __init__(self, trueLabels, PredictedLabes):
        self.trueLabels = trueLabels
        self.PredictedLabes = PredictedLabes
    
    # True positive rating. 
    def truePositiveRate(self):
        truePositives = np.sum((self.trueLabels == 1) & (self.PredictedLabes == 1))
        actualPositives = np.sum(self.trueLabels == 1)

        return (truePositives) / (actualPositives) if actualPositives != 0 else 0

    # False positive rating. 
    def falsePositiveRate(self):
        falsePositives = np.sum((self.trueLabels == 0) & (self.PredictedLabes == 1))
        actualNegatives = np.sum(self.trueLabels == 0)

        return (falsePositives) / (actualNegatives) if actualNegatives != 0 else 0
    
    # F1 score.
    def f1Score(self):
        precision = self.truePositiveRate()
        recall = self.truePositiveRate()

        return (2 * (precision * recall)) / (precision + recall) if (precision + recall) != 0 else 0

    # Accuracy of the model. 
    def Accuracy(self):
        accuracy = np.mean(self.trueLabels == self.PredictedLabes)

        return accuracy
    
    # Confusion matrix. 
    def ConfusionMatrix(self):
        uniqueLabels = np.unique(np.concatenate((self.trueLabels, self.PredictedLabes)))
        numberOfClasses = len(uniqueLabels)
        confusionMatrix = np.zeros((numberOfClasses, numberOfClasses), dtype = int)

        for i, true_Label in enumerate(uniqueLabels):
            true_Indices = (self.trueLabels == true_Label)
            for j, predicted_Label in enumerate(uniqueLabels):
                predicted_Indices = (self.PredictedLabes == predicted_Label)
                confusionMatrix[i, j] = np.sum(true_Indices & predicted_Indices)
        
        return confusionMatrix
