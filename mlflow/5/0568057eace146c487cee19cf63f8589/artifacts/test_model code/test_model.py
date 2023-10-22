import pickle

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("test_model")

with mlflow.start_run():
    data = pd.read_csv('/home/pyretttt/repos/mlops3/datasets/test.csv', sep=',')
    with open('/home/pyretttt/repos/mlops3/models/clf.pickle', 'rb') as f:
        model = pickle.load(f)

        score = model.score(data.iloc[:, :-1], data.iloc[:, -1])
        print('score=', score)

        mlflow.log_artifact(local_path='/home/pyretttt/repos/mlops3/scripts/test_model.py',
                artifact_path="test_model code")
        mlflow.log_metric("score", score)
        mlflow.end_run()


