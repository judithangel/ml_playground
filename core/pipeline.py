from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score, \
    adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def train_and_eval(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {"model": model, "accuracy": acc, "predictions": y_pred, "test-set": X_test}

def cluster_and_eval(X, model, y_true=None):
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)
    if y_true is not None:
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
        return {"model": model, "predictions": labels, "test-set": X, "n_clusters": n_clusters,
                "silhouette": silhouette, "calinski": calinski, "davies": davies, "ari": ari, "nmi": nmi}
    return {"model": model, "predictions": labels, "test-set": X, "n_clusters": n_clusters, "silhouette": silhouette,
            "calinski": calinski, "davies": davies}

def pca_project(X, n_components=2):
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca
