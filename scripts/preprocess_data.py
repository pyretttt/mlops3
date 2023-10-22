from sklearn.preprocessing import LabelEncoder
import pandas as pd

data_frame = pd.read_csv("/home/pyretttt/repos/mlops3/datasets/iris.csv", sep=',')
data_frame['variety'] = LabelEncoder().fit_transform(data_frame['variety'])

data_frame.to_csv("/home/pyretttt/repos/mlops3/datasets/data_prepared.csv", index=False)

