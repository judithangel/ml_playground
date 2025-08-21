from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

params_KNN = {
    "K": {
        "type": "int",
        "min": 1,
        "max": 100,
        "default": 5,
        "step": 1
    }
}

params_SVM = {
    "C": {
        "type": "float",
        "min": 0.01,
        "max": 10.0,
        "default": 1,
        "step": 0.01
    }
}

params_RF = {
    "max_depth": {
        "type": "int",
        "min": 2,
        "max": 15,
        "default": 5,
        "step": 1
    },
    "n_estimators": {
        "type": "int",
        "min": 1,
        "max": 100,
        "default": 10,
        "step": 1
    }
}

def list_models():
    return ["KNN", "SVM", "Random Forest"]

def get_param_space(model_name):
    if model_name == "KNN":
        return params_KNN
    elif model_name == "SVM":
        return params_SVM
    elif model_name == "Random Forest":
        return params_RF
    else:
        raise ValueError("Unknown model name")

def build_model(name, params):
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=params["K"])
    elif name == "SVM":
        return SVC(C=params["C"])
    elif name == "Random Forest":
        return RandomForestClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"])
    else:
        raise ValueError("Unknown model name")

params_KMeans = {
    "n_clusters": {
        "type": "int",
        "min": 2,
        "max": 10,
        "default": 3,
        "step": 1
    },
    "init": {
        "type": "choice",
        "choices": ["k-means++", "random"],
        "default": "k-means++"
    },
    "n_init": {
        "type": "int",
        "min": 1,
        "max": 10,
        "default": 10,
        "step": 1
    }
}

params_DBSCAN = {
    "eps": {
        "min": 0.1,
        "max": 1.0,
        "default": 0.5,
        "step": 0.1
    },
    "min_samples": {
        "min": 1,
        "max": 10,
        "default": 5,
        "step": 1
    }
}

params_AC = {
    "n_clusters": {
        "min": 2,
        "max": 10,
        "default": 3,
        "step": 1
    },
    "linkage": {
        "values": ["ward", "complete", "average"],
        "default": "ward"
    }
}

def list_clusters():
    return ["KMeans", "DBSCAN", "Agglomerative Clustering"]

def get_cluster_param_space(cluster_name):
    if cluster_name == "KMeans":
        return params_KMeans
    elif cluster_name == "DBSCAN":
        return params_DBSCAN
    elif cluster_name == "Agglomerative Clustering":
        return params_AC
    else:
        raise ValueError("Unknown cluster name")

def build_cluster(name, params):
    if name == "KMeans":
        return KMeans(n_clusters=params["n_clusters"], init=params["init"], n_init=params["n_init"])
    elif name == "DBSCAN":
        return DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    elif name == "Agglomerative Clustering":
        return AgglomerativeClustering(n_clusters=params["n_clusters"], linkage=params["linkage"], affinity=params["affinity"])
    else:
        raise ValueError("Unknown cluster name")
