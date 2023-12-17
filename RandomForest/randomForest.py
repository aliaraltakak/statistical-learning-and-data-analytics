# Import the libraries. 
import numpy as np

class createNode:
    
    # Define a node class to use in the decision trees/random forest. 
    def __init__(self, Feature=None, Threshold=None, createLeft=None, createRight=None, *, Val=None):
        self.Feature = Feature
        self.Threshold = Threshold
        self.createLeft = createLeft
        self.createRight = createRight
        self.Val = Val
    
    # A method that decides if a node is a leaf node or not.
    def leafNodeDecider(self):
        if self.Val is not None:
            return True
        else:
            return False

class decisionTree:
    
    # Initialize a decision tree classifier. 
    def __init__(self, minSampleSplit=2, maximumDepth=100, featureCount=None):
        self.minSampleSplit = minSampleSplit
        self.maximumDepth = maximumDepth
        self.featureCount = featureCount
        self.Root = None

    # Main method to train the decision tree. 
    def fitData(self, Features, Labels):
        if self.featureCount is None:
            self.featureCount = Features.shape[1]
        else:
            self.featureCount = min(Features.shape[1], self.featureCount)
        self.Root = self.growTree(Features, Labels)

    # A method that grows the decision tree recursively. 
    def growTree(self, Features, Labels, Depth=0):
        sampleCount, featureCount = Features.shape
        labelCount = len(np.unique(Labels))

        if (Depth >= self.maximumDepth or labelCount == 1 or sampleCount < self.minSampleSplit):
            leafVal = self.mostCommonLabel(Labels)
            
            return createNode(Val=leafVal)

        featureIndices = np.random.choice(featureCount, self.featureCount, replace=False)
        bestFeature, bestThreshold = self.bestSplit(Features, Labels, featureIndices)

        leftIndices, rightIndices = self.split(Features[:, bestFeature], bestThreshold)
        createLeft = self.growTree(Features[leftIndices, :], Labels[leftIndices], Depth + 1)
        createRight = self.growTree(Features[rightIndices, :], Labels[rightIndices], Depth + 1)
        
        return createNode(Feature=bestFeature, Threshold=bestThreshold, createLeft=createLeft, createRight=createRight)

    # A method that is used to calculate and decide the best split. 
    def bestSplit(self, Features, Labels, featureIndices):
        bestGain = -1
        splitIndex, splitThreshold = None, None

        for featureIndex in featureIndices:
            FeatureColumn = Features[:, featureIndex]
            thresholds = np.unique(FeatureColumn)

            for threshold in thresholds:
                gain = self.informationGain(Labels, FeatureColumn, threshold)

                if gain > bestGain:
                    bestGain = gain
                    splitIndex = featureIndex
                    splitThreshold = threshold

        return splitIndex, splitThreshold

    # A method that splits the dataset depending on a threshold.
    def split(self, FeatureColumn, splitThreshold):
        leftIndices = np.argwhere(FeatureColumn <= splitThreshold).flatten()
        rightIndices = np.argwhere(FeatureColumn > splitThreshold).flatten()
        return leftIndices, rightIndices

    # A method that calculates the information gain on a split. 
    def informationGain(self, Labels, FeatureColumn, threshold):
        parentEntropy = self.Entropy(Labels)
        leftIndices, rightIndices = self.split(FeatureColumn, threshold)

        if len(leftIndices) == 0 or len(rightIndices) == 0:
            return 0

        n = len(Labels)
        nL, nR = len(leftIndices), len(rightIndices)
        eL, eR = self.Entropy(Labels[leftIndices]), self.Entropy(Labels[rightIndices])
        childEntropy = (nL / n) * eL + (nR / n) * eR

        infoGain = (parentEntropy - childEntropy)
        
        return infoGain

    # A method that calculates the entropy, a parameter that is used in the process of information gain. 
    def Entropy(self, Labels):
        # Convert labels to integers
        integerLabels = Labels.astype(int)

        # Compute histogram
        hist = np.bincount(integerLabels)

        # Continue with the original entropy calculation
        ps = hist / len(Labels)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    # A method that determines the most common label on the dataset. 
    def mostCommonLabel(self, Labels):
        label_counts = {}
        most_common_elem = None
        highest_count = 0

        for label in Labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

            if label_counts[label] > highest_count:
                most_common_elem = label
                highest_count = label_counts[label]

        return most_common_elem

    # A method that computes predictions. 
    def Predictor(self, Features):
        # Create an empty list to store the predictions
        predictions = []

        # Iterate over each feature in the Features array
        for feature in Features:
            # For each feature, traverse the tree and get the prediction
            prediction = self.traverseTree(feature, self.Root)
            # Append the prediction to the predictions list
            predictions.append(prediction)

        # Convert the list of predictions to a numpy array and return it
        return np.array(predictions)
        
    # A method that traverses the decision tree recursively to find the class label for a feature.
    def traverseTree(self, feature, node):
        if node.leafNodeDecider():
            
            return node.Val

        if feature[node.Feature] <= node.Threshold:
            
            return self.traverseTree(feature, node.createLeft)
        else:
            
            return self.traverseTree(feature, node.createRight)




class randomForestClassifier:
    
    # Define a random forest classifier.
    def __init__(self, numberofTrees = 15, maximumDepth = 15, minSampleSplit = 2, featureCount = None):
        
        self.numberofTrees = numberofTrees
        self.maximumDepth = maximumDepth
        self.minSampleSplit = minSampleSplit
        self.featureCount = featureCount
        self.decisionTrees = []

    def fitData(self, Features, Labels):
        
        self.decisionTrees = []
        
        for _ in range(self.numberofTrees):
            
            # Initialize a decision tree. 
            tree = decisionTree(maximumDepth = self.maximumDepth,
                            minSampleSplit = self.minSampleSplit,
                            featureCount = self.featureCount)
            
            X_sample, y_sample = self.sampleBootstrap(Features, Labels)
            tree.fitData(X_sample, y_sample)
            self.decisionTrees.append(tree)

    def sampleBootstrap(self, Features, Labels):
        
        sampleCount = Features.shape[0]
        idxs = np.random.choice(sampleCount, sampleCount, replace=True)
        
        return Features[idxs], Labels[idxs]

    def mostCommonLabel(self, Labels):
        label_counts = {}
        most_common_elem = None
        highest_count = 0

        for label in Labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

            if label_counts[label] > highest_count:
                most_common_elem = label
                highest_count = label_counts[label]

        return most_common_elem


    def Predictor(self, X):
        
        predictions = np.array([tree.Predictor(X) for tree in self.decisionTrees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.mostCommonLabel(pred) for pred in tree_preds])
        return predictions