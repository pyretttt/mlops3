import pickle

import pandas as pd

data = pd.read_csv('/home/pyretttt/repos/mlops3/datasets/test.csv', sep=',')
with open('/home/pyretttt/repos/mlops3/models/clf.pickle', 'rb') as f:
    model = pickle.load(f)

    score = model.score(data.iloc[:, :-1], data.iloc[:, -1])
    print('score=', score)


