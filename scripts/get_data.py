import sys

import requests

data = requests.get("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")

with open("/home/pyretttt/repos/mlops3/datasets/iris.csv", "w") as f:
    f.write(data.text)


