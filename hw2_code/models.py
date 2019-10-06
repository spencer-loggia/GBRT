"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np

class RegressionTree(object):
    def __init__(self, nfeatures, max_depth):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.feature = -1
        self.theta = -1
        self.right = -1
        self.left = -1

    def scoreSplit(self, feature, threshold, X, y):
        featureCol = np.transpose(X)[feature]
        if type(threshold) != np.float64:
            print('bad')
        R = np.argwhere(featureCol > threshold)
        Rmean = np.mean(featureCol[R])
        L = np.argwhere(featureCol <= threshold)
        Lmean = np.mean(featureCol[L])
        ssR = 0
        ssL = 0
        for r in R:
            ssR += np.square(y[r[0]] - Rmean)
        for l in L:
            ssL += np.square(y[l[0]] - Lmean)
        return {'score': ssR + ssL,
                'left': L,
                'right': R}

    def bestSplit(self, X, y):
        scores = []
        min_score = 999999999999999
        min_theta = 999999999999999
        feature_split = -1
        minR = []
        minL = []
        for f in range(0, self.num_input_features):
            for t in X:
                if type(t[f]) != np.float64:
                    print('vbad')
                cur_score = self.scoreSplit(f, t[f], X, y)
                if cur_score['score'] < min_score:
                    min_score = cur_score['score']
                    min_theta = t[f]
                    feature_split = f
                    minR = cur_score['right']
                    minL = cur_score['left']
        self.feature = feature_split
        self.theta = min_theta
        return [minL, minR]

    def split(self, X, y):
        data = self.bestSplit(X, y)
        LX = []
        Ly = []
        RX = []
        Ry = []
        for i in data[0]:
            LX.append(X[i[0]])
            Ly.append(y[i[0]])
        for i in data[1]:
            RX.append(X[i[0]])
            Ry.append(y[i[0]])
                #use dictionary
        return {'leftX': np.array(LX),
                'lefty': np.array(Ly),
                'rightX': np.array(RX),
                'righty': np.array(Ry)}

    def fit(self, *, X, y, base=True):
        """ Fit the model.
                   Args:
                   X: A of floats with shape [num_examples, num_features].
                   y: An array of floats with shape [num_examples].
                   max_depth: An int representing the maximum depth of the tree
        """
        newData = self.split(X, y)
        if self.max_depth > 0 and len(X) > 1:
            self.left = RegressionTree(self.num_input_features, self.max_depth - 1)
            self.left.fit(X=newData['leftX'], y=newData['lefty'])
            self.right = RegressionTree(self.num_input_features, self.max_depth - 1)
            self.right.fit(X=newData['rightX'], y=newData['righty'])
            return self
        else:
            self.left = np.mean(newData['lefty'])
            self.right = np.mean(newData['righty'])

    def predict(self, X):
        if X[self.feature] <= self.theta:
            if type(self.left) == np.float64:
                return self.left
            else:
                return self.left.predict(X)
        elif X[self.feature] > self.theta:
            if type(self.right) == np.float64:
                return self.right
            else:
                return self.right.predict(X)



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
