"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np

class RegressionTree(object):
    def __init__(self, nfeatures, max_depth, usedfeatures=np.zeros(0)):
        self.num_input_features = nfeatures
        if len(usedfeatures) > 0:
            self.used_features = usedfeatures
        else:
            self.used_features = np.zeros(nfeatures, dtype=0)
        self.max_depth = max_depth
        self.feature = -1
        self.right = -1
        self.left = -1
        self.X = -1
        self.y = -1

    def scoreSplit(self, feature, threshold):
        score = 0
        return score

    def bestSplit(self):
        scores = []
        max_score = -1
        max_theta = -1
        feature_split = -1
        for f in range(0, self.num_input_features):
            if self.used_features[f] == 0:
                fmax_score = -1
                fmax_theta = -1
                for t in self.X:
                    cur_score = self.scoreSplit(f, t[f])
                    if cur_score > fmax_score:
                        fmax_score = cur_score
                        fmax_theta = t[f];
                if fmax_score > max_score:
                    max_score = fmax_score
                    max_theta = fmax_theta
                    feature_split = f
        return [feature_split, max_theta]

    def split(self):
        stats = self.bestSplit()
        rightX = []
        righty = []
        leftX = []
        lefty = []
        theta = stats[1]
        feature = stats[0]
        temp = np.transpose(self.X)
        featureCol = temp[self.feature]
        for i in featureCol:
            if i <= theta:
                leftX.append(self.X[i])
                lefty.append(self.y[i])
            else:
                rightX.append(self.X[i])
                righty.append(self.y[i])

                #use dictionary
        return {'leftX': leftX,
                'lefty': lefty,
                'rightX': rightX,
                'righty': righty}

    def fit(self, *, X, y):
        """ Fit the model.
                   Args:
                   X: A of floats with shape [num_examples, num_features].
                   y: An array of floats with shape [num_examples].
                   max_depth: An int representing the maximum depth of the tree
        """
        self.X = self.np.array(X)
        self.y = np.array(y)
        newData = self.split()
        self.left = RegressionTree(self.num_input_features, self.max_depth - 1, usedfeatures=self.used_features)
        self.left.fit(X=newData['leftX'], y=newData['lefty'])
        self.right = RegressionTree(self.num_input_features, self.max_depth - 1, usedfeatures=self.used_features)
        self.right.fit(X=newData['rightX'], y=newData['righty'])
        return self


    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")



class GradientBoostedRegressionTree(object):
    def __init__(self, nfeatures, max_depth, n_estimators, regularization_parameter):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.regularization_parameter = regularization_parameter
    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
                n_estimators: An int representing the number of regression trees to iteratively fit
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        # TODO: Implement this!
        raise Exception("You must implement this method!")
