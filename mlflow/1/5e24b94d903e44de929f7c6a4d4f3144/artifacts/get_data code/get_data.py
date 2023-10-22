import sys

import requests
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("get_model")

with mlflow.start_run():
    data = requests.get("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")

    with open("/home/pyretttt/repos/mlops3/datasets/iris.csv", "w") as f:
        f.write(data.text)
        mlflow.log_artifact(local_path="/home/pyretttt/repos/mlops3/scripts/get_data.py",
                                    artifact_path="get_data code")
        mlflow.end_run()

