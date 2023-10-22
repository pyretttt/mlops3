import pickle

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('/home/pyretttt/repos/mlops3/datasets/train.csv', sep=',')
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
X = data.iloc[:,[0,1,2,3]]
y = data.iloc[:,4]

clf.fit(X, y)

with open('/home/pyretttt/repos/mlops3/models/clf.pickle', 'wb') as f:
    pickle.dump(clf, f)
