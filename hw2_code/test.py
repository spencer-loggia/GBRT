import numpy as np
import models
from data import load_data

y = [4,2,6,1,2]
X = [[4,1,5,4,1], [4,1,4,6,4], [1,4,7,2,3], [4,6,7,2,5]]

test = models.RegressionTree(nfeatures=4, max_depth=2)
test = test.fit(X=X, y=y)
