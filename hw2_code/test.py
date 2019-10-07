import numpy as np
import models
from data import load_data

y = [4,2,6,1,2,3]
X = [[4,1,5,4,1], [4,1,4,6,4], [4,1,1,6,5], [1,4,7,2,3], [4,6,7,2,5], [1,1,1,2,6]]
X, y = load_data('../data/train.txt')

test = models.RegressionTree(nfeatures=37, max_depth=3)
test.fit(X=X, y=y)
# pred = test.predict(X[2])

test.traverse()

# devX, devy = load_data('../data/train_sm.txt')
# pred = test.predict(devX)
# print("pred: " + str(pred))
#
#
