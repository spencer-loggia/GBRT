"""
Keep model implementations in here.

This file is where you will write most of your code!
"""

import numpy as np
from collections import OrderedDict


class RegressionTree(object):

    # to map array pf feature, threshold, sortedXs, sortedYs to dictionary storing relevent info
    calculated = {}

    def __init__(self, nfeatures, max_depth):
        self.num_input_features = nfeatures
        self.max_depth = max_depth
        self.feature = -1
        self.theta = -1
        self.right = -1
        self.left = -1  # used to make scores comparable across features

    def scoreSplit(self, feature, t_arg, Xs, y):
        # Xs and y are sorted in best split
        R = Xs[t_arg:len(y)]
        yr = y[t_arg:len(y)]
        L = Xs[0:t_arg]
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
            calculated = tuple([tuple([f]), X.data.tobytes()])
            data = np.vstack([np.transpose(X), y])
            data = np.transpose(data)
            if str(calculated) in RegressionTree.calculated:
                col = RegressionTree.calculated[calculated]
            else:
                col = data[:, f].argsort(kind='mergesort')
                RegressionTree.calculated.update({calculated: col})

            try:
                data = data[col]
            except IndexError:
                print('DATA len: ' + str(len(data)))
                print('COL len' + str(len(col)))

            Xs = np.transpose(data)[0:len(X[0]), :]
            Xs = np.transpose(Xs)
            ys = np.transpose(data)[len(data[0])-1]
            prev_theta = -1
            unique_thetas = list(OrderedDict.fromkeys(Xs[:, f]))
            index = 0
            for t in range(0, len(unique_thetas)):
                while Xs[index][f] != unique_thetas[t]:
                    index += 1
                cur_score = self.scoreSplit(f, index, Xs, ys)
                index += 1
                if type(cur_score) != int and cur_score['score'] < min_score and \
                        cur_score['leftX'].size > 0 and cur_score['rightX'].size > 0:
                    min_output = cur_score
                    min_score = min_output['score']
                    min_theta = unique_thetas[t]
                    feature_split = f

        self.feature = feature_split
        self.theta = min_theta
        if type(min_output['rightX']) == list or type(min_output['leftX']) == list \
                or min_output['rightX'].size < 2 or min_output['leftX'].size < 2:
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
        if self.max_depth > 1 and newData['lefty'].size > 1 and newData['variance']:
            self.left = RegressionTree(self.num_input_features, self.max_depth - 1)
            self.left.fit(X=newData['leftX'], y=newData['lefty'], base=False)
        else:
            self.left = np.mean(newData['lefty'])

        if self.max_depth > 1 and newData['righty'].size > 1 and newData['variance']:
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
        prediction = np.zeros(len(X))
        for t in self.forest:
            pred1 = np.array(t.predict(X))
            pred2 = self.regularization_parameter * pred1
            prediction += pred2
        return prediction

