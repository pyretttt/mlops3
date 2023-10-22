import pickle

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("train_data")

with mlflow.start_run():

    data = pd.read_csv('/home/pyretttt/repos/mlops3/datasets/train.csv', sep=',')
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    X = data.iloc[:,[0,1,2,3]]
    y = data.iloc[:,4]

    clf.fit(X, y)

    mlflow.sklearn.log_model(clf,
            artifact_path="lr",
            registered_model_name="lr")
    mlflow.log_artifact(local_path="/home/pyretttt/repos/mlops3/scripts/train_model.py",
            artifact_path="train_model code")
    mlflow.end_run()

    with open('/home/pyretttt/repos/mlops3/models/clf.pickle', 'wb') as f:
        pickle.dump(clf, f)

