from sklearn.preprocessing import LabelEncoder
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow.set_experiment("preprocess_data")

with mlflow.start_run():
    data_frame = pd.read_csv("/home/pyretttt/repos/mlops3/datasets/iris.csv", sep=',')
    data_frame['variety'] = LabelEncoder().fit_transform(data_frame['variety'])

    data_frame.to_csv("/home/pyretttt/repos/mlops3/datasets/data_prepared.csv", index=False)

    mlflow.log_artifact(local_path="/home/pyretttt/repos/mlops3/scripts/preprocess_data.py",
                                    artifact_path="preprocess_data code")
    mlflow.end_run()

