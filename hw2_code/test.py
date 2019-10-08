import numpy as np
import models
from data import load_data
import pickle

y = [-3, 6, 8]
X = [[0,3,1], [4,2,3], [5,1,2]]

X, y = load_data('../data/train.txt')
y=np.array(y)
X=np.array(X)


def train(algorithm='gradient-boosted-regression-tree'):
    """ Fit a model's parameters given the parameters specified in args.
    """
    X, y = load_data('../data/train.txt')

    # Initialize appropriate algorithm
    if algorithm == 'regression-tree':
        model = models.RegressionTree(nfeatures=37, max_depth=3)
    elif algorithm == 'gradient-boosted-regression-tree':
        model = models.GradientBoostedRegressionTree(nfeatures=37, max_depth=3, n_estimators=4, regularization_parameter=.2)
    else:
        raise Exception("Algorithm argument not recognized")

    model.fit(X=X, y=y)

    # Save the model
    pickle.dump(model, open('test.model', 'wb'))

def test():
    """ Make predictions over the input test dataset, and store the predictions.
    """
    # load dataset and model
    X, observed_y = load_data('../data/dev.txt')

    model = pickle.load(open('test.model', 'rb'))

    # predict labels for dataset
    preds = model.predict(X)

    # output model predictions
    np.savetxt('test.predictions', preds, fmt='%s')


train()
test()

#
#
