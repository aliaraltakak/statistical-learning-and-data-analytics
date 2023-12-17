# Import libraries. 
import numpy as np 

class gaussianNaiveBayes:

    # Initialize the required variables for Gaussian Naive Bayes Classifier. 
    def fitData(self, features, labels):
        self.classes = np.unique(labels)
        self.classPriors = {}
        self.classMeans = {}
        self.classVariances = {}

        # Compute the required variables. 
        for count in self.classes:
            featureCount = features[labels == count]
            self.classPriors[count] = len(featureCount) / len(features)
            self.classMeans[count] = np.mean(featureCount, axis = 0)
            self.classVariances[count] = np.var(featureCount, axis = 0)
    
    # Calculate the likelihood of instances using the normal distribution formula. 
    def likelihoodCalculation(self, features, mean, variance):
        exponential = np.exp(-((features - mean) ** 2) / (2 * variance))
        return (1 / (np.sqrt(2 * np.pi * variance))) * exponential
    
    # Compute the posterior probability of instances of the dataset. 
    def instancePrediction(self, features):
        posteriorData = []

        for count in self.classes:
            priorProb = np.log(self.classPriors[count])
            likelihood = np.sum(np.log(self.likelihoodCalculation(features, self.classMeans[count], self.classVariances[count])))
            posterior = (priorProb) + (likelihood)
            posteriorData.append(posterior)
        
        return self.classes[np.argmax(posteriorData)]
    
    # Compute predictions. 
    def Predictor(self, features):
        return np.apply_along_axis(self.instancePrediction, axis = 1, arr = features)

