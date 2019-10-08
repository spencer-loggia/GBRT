import numpy as np
import models
from data import load_data

y = [-3, 6, 8]
X = [[0,3,1], [4,2,3], [5,1,2]]


test = models.RegressionTree(nfeatures=3, max_depth=2)
test.fit(X=X, y=y)
# pred = test.predict(X[2])

test.traverse()

# devX, devy = load_data('../data/train_sm.txt')
# pred = test.predict(devX)
# print("pred: " + str(pred))
#
#
