import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_frame = pd.read_csv('/home/pyretttt/repos/mlops3/datasets/data_prepared.csv', sep=',')
X_train, X_test, y_train, y_test = train_test_split(data_frame.iloc[:, :-1], data_frame.iloc[:, -1], test_size = 0.2, random_state=42)

pd.concat((X_train, y_train), axis=1).to_csv('/home/pyretttt/repos/mlops3/datasets/train.csv', index=False)
pd.concat((X_test, y_test), axis=1).to_csv('/home/pyretttt/repos/mlops3/datasets/test.csv', index=False)


