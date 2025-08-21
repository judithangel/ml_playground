import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine

def list_datasets():
    return ["breast_cancer", "diabetes", "iris", "wine"]

def load_dataset(name):
    if name == "breast_cancer":
        data = load_breast_cancer()
    elif name == "diabetes":
        data = load_diabetes()
    elif name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    X = data.data
    y = data.target
    return X, y

def load_csv(file, target=None, drop_non_num=True):
    df = pd.read_csv(file)
    if drop_non_num:
        non_numerical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        df.drop(columns=non_numerical_cols, inplace=True)
    df.dropna(inplace=True)
    if target is None:
        y = None
        X = df.copy()
    else:
        y = df[target]
        X = df.drop(columns=[target])
    return X, y