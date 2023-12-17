# Import the libraries. 
import numpy as np

class SupportVectorMachine:
    
    def __init__(self, learningRate = 0.001, lambdaParameter = 0.05, iterationCount = 1500):
        
        self.learningRate = learningRate # Learning rate of the model.
        self.lambdaParameter = lambdaParameter # Regularization parameter. 
        self.iterationCount = iterationCount # Number of iterations for training. 
        self.Weight = None # Weight vector. 
        self.Bias = None # Bias term. 
        
    # Main method to train the model on the training data. 
    def fit(self, feature, label):
        
        # Obtain the number of samples and features of the dataset. 
        noSamples, noFeatures = feature.shape
        
        # Convert the binary labels of (0,1) to (-1,1) for separation. 
        label_ = np.where(label <= 0, -1, 1)

        # Set the weights and bias to zero. 
        self.Weight = np.zeros(noFeatures)
        self.Bias = 0
        
        # Training by stochastic gradient descent algorithm and iteration. 
        for _ in range(self.iterationCount):
            for idx, x_i in enumerate(feature):
                condition = label_[idx] * (np.dot(x_i, self.Weight) - self.Bias) >= 1
                
                # Update rules for weights and bias. 
                if condition:
                    self.Weight -= self.learningRate * (2 * self.lambdaParameter * self.Weight)
                else:
                    self.Weight -= self.learningRate * (2 * self.lambdaParameter * self.Weight - np.dot(x_i, label_[idx]))
                    self.Bias -= self.learningRate * label_[idx]

    # Main predictor method of the Support Vector Machine Classifier model. 
    def predict(self, X):
        
        # Calculation of the decision boundary. 
        approx = np.dot(X, self.Weight) - self.Bias
        # Return the predictions as 0 and 1 instead of -1 and 1 for better fit.
        return np.where(np.sign(approx) == -1, 0, 1)




