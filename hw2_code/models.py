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
        self.normy = 1  # to make scores comparable across features

    def scoreSplit(self, feature, threshold, X, y):
        featureCol = np.transpose(X)[feature]
        norm = np.linalg.norm(featureCol)
        if norm == 0:
            return - 1
        featureCol = featureCol / norm
        threshold = threshold / norm
        R = np.argwhere(featureCol > threshold)
        L = np.argwhere(featureCol <= threshold)
        ssR = np.sum(np.square(y[R] / self.normy - np.mean(featureCol[R])))
        ssL = np.sum(np.square(y[L] / self.normy - np.mean(featureCol[L])))
        return {'score': ssR + ssL,
                'left': L,
                'right': R}

    def bestSplit(self, X, y):
        scores = []
        min_score = 9999999999999999
        min_theta = 9999999999999999
        feature_split = -1
        minR = []
        minL = []
        for f in range(0, self.num_input_features):
            for t in X:
                cur_score = self.scoreSplit(f, t[f], X, y)
                if type(cur_score) != int and cur_score['score'] < min_score and \
                        cur_score['left'].size > 0 and cur_score['right'].size > 0:
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
        variance = True
        if type(data[0]) == list or data[0].size == 0 or data[1].size == 0:
            variance = False
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
                'righty': np.array(Ry),
                'variance': variance}

    def fit(self, *, X, y, base=True):
        """ Fit the model.
                   Args:
                   X: A of floats with shape [num_examples, num_features].
                   y: An array of floats with shape [num_examples].
                   max_depth: An int representing the maximum depth of the tree
        """
        self.normy = np.linalg.norm(y)
        newData = self.split(X, y)
        if self.max_depth > 0 and len(X) > 1 and newData['variance']:
            self.left = RegressionTree(self.num_input_features, self.max_depth - 1)
            self.left.fit(X=newData['leftX'], y=newData['lefty'])
            self.right = RegressionTree(self.num_input_features, self.max_depth - 1)
            self.right.fit(X=newData['rightX'], y=newData['righty'])
            return self
        else:
            self.left = np.mean(newData['lefty'])
            self.right = np.mean(newData['righty'])

    def predictOne(self, X):
        if X[self.feature] <= self.theta:
            if type(self.left) == np.float64:
                return self.left
            else:
                return self.left.predictOne(X)
        elif X[self.feature] > self.theta:
            if type(self.right) == np.float64:
                return self.right
            else:
                return self.right.predictOne(X)

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self.predictOne(x)
            predictions.append(pred)
        return predictions

    def traverse(self, level = 0):
        print(str(level) + " :: feature: " + str(self.feature) + "  theta: " + str(self.theta))
        if type(self.left) == np.float64:
            print(str(level) + ": val: " + str(self.left))
            return 0
        else:
            self.left.traverse(level+1)
        if type(self.right) == np.float64:
            print(str(level) + ": val: " + str(self.right))
            return 0
        else:
            self.right.traverse(level+1)


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
