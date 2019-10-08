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
        self.left = -1 # used to make scores comparable across features

    def scoreSplit(self, feature, threshold, Xs, y):
        Xs = np.transpose(Xs)
        t_arg = np.searchsorted(Xs[feature], threshold)
        R = np.transpose(Xs)[t_arg:len(y)]
        yr = y[t_arg:len(y)]
        L = np.transpose(Xs)[0:t_arg]
        yl = y[0:t_arg]
        if R.size == 0:
            ssR = 0
        else:
            ssR = np.sum(np.square(yr - np.mean(yr)))
        if L.size == 0:
            ssL = 0
        else:
            ssL = np.sum(np.square(yl - np.mean(yl)))
        return {'score': ssR + ssL,
                'leftX': L,
                'rightX': R,
                'lefty': yl,
                'righty': yr}

    def bestSplit(self, X, y):
        min_score = 9999999999999999
        min_theta = 9999999999999999
        feature_split = -1
        min_output = {}
        variance = True
        for f in range(0, self.num_input_features):
            data = np.vstack([np.transpose(X), y])
            data = np.transpose(data)
            col = data[:, f].argsort(kind='mergesort')
            data = data[col]
            Xs = np.transpose(data)[0:len(X[0]), :]
            Xs = np.transpose(Xs)
            ys = np.transpose(data)[len(data[0])-1]
            for t in range(0, len(X)):
                cur_score = self.scoreSplit(f, X[t][f], Xs, ys)
                if type(cur_score) != int and cur_score['score'] < min_score and \
                        cur_score['leftX'].size > 0 and cur_score['rightX'].size > 0:
                    min_output = cur_score
                    min_score = min_output['score']
                    min_theta = X[t][f]
                    feature_split = f
        self.feature = feature_split
        self.theta = min_theta
        if type(min_output['rightX']) == list or type(min_output['leftX']) == list \
                or min_output['rightX'].size == 0 or min_output['leftX'].size == 0:
            variance = False
        min_output.update({'variance': variance})
        return min_output

    def fit(self, *, X, y, base=True):
        """ Fit the model.
                   Args:
                   X: A of floats with shape [num_examples, num_features].
                   y: An array of floats with shape [num_examples].
                   max_depth: An int representing the maximum depth of the tree
        """
        newData = self.bestSplit(X, y)
        if self.max_depth > 0 and newData['lefty'].size > 1 and newData['variance']:
            self.left = RegressionTree(self.num_input_features, self.max_depth - 1)
            self.left.fit(X=newData['leftX'], y=newData['lefty'], base=False)
        else:
            self.left = np.mean(newData['lefty'])

        if self.max_depth > 0 and newData['righty'].size > 1 and newData['variance']:
            self.right = RegressionTree(self.num_input_features, self.max_depth - 1)
            self.right.fit(X=newData['rightX'], y=newData['righty'], base=False)
        else:
            self.right = np.mean(newData['righty'])

    def predictOne(self, X):
        if X[self.feature] < self.theta:
            if type(self.left) == np.float64:
                return self.left
            else:
                return self.left.predictOne(X)
        elif X[self.feature] >= self.theta:
            if type(self.right) == np.float64:
                return self.right
            else:
                return self.right.predictOne(X)

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self.predictOne(x)
            predictions.append(pred)
        return np.array(predictions)

    def traverse(self, level = 0):
        print(str(level) + " :: feature: " + str(self.feature) + "  theta: " + str(self.theta))
        if type(self.left) == np.float64:
            print(str(level) + ": val: " + str(self.left))
        else:
            self.left.traverse(level+1)

        if type(self.right) == np.float64:
            print(str(level) + ": val: " + str(self.right))
        else:
            self.right.traverse(level+1)
        return 0


class GradientBoostedRegressionTree(object):
    def __init__(self, nfeatures, max_depth, n_estimators, regularization_parameter):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.regularization_parameter = regularization_parameter
        self.forest = []
        self.initial_prediction = 0
        self.prediction = 0

    def calculate_residuals(self, predictions, y):
        return np.array(y - predictions)

    def fit(self, *, X, y):
        """ Fit the model.
                Args:
                X: A of floats with shape [num_examples, num_features].
                y: An array of floats with shape [num_examples].
                max_depth: An int representing the maximum depth of the tree
                n_estimators: An int representing the number of regression trees to iteratively fit
        """
        for t in range(0, self.n_estimators):
            prediction = []
            if t == 0:
                prediction = self.initial_prediction = np.ones(y.size) * np.mean(y)
            else:
                prediction = self.prediction

            resi = self.calculate_residuals(prediction, y)
            tree = RegressionTree(nfeatures=self.num_input_features, max_depth=self.max_depth)
            tree.fit(X=X, y=resi)
            self.add_prediction(X, tree)
            self.forest.append(tree)

    def add_prediction(self, X, tree):
        p = np.array(tree.predict(X))
        self.prediction += self.regularization_parameter * p

    def predict(self, X):
        """ Predict.
        Args:
                X: A  matrix of floats with shape [num_examples, num_features].

        Returns:
                An array of floats with shape [num_examples].
        """
        prediction = self.initial_prediction
        for t in self.forest:
            prediction += self.regularization_parameter * t.predict(X)
        return prediction

