import streamlit as st
from core.datasets import load_dataset, load_csv, list_datasets
from core.models import list_models, get_param_space, build_model, list_clusters, get_cluster_param_space, \
      build_cluster
from core.pipeline import pca_project, train_and_eval, cluster_and_eval
import plotly.express as px

# Seiteneinstellungen für Streamlit
st.set_page_config(page_title="ML Playground", layout="wide")
st.title("Streamlit example")

problem_type = st.sidebar.radio("Select type of problem", ["Classification", "Clustering"])
data_source = st.sidebar.selectbox("Select data source", ["Upload CSV", "Use Sample Data"])
if data_source == "Upload CSV":
    has_target = st.checkbox("Has target?")
    if has_target:
        target = st.text_input("Target column:")
    else:
        target = None
    data = st.file_uploader("Upload CSV", type=["csv"])
    X, y = load_csv(data, target=target)
elif data_source == "Use Sample Data":
    dataset_name = st.sidebar.selectbox("Select dataset", list_datasets())
    X, y = load_dataset(dataset_name)
if problem_type == "Classification":
    model = st.sidebar.selectbox("Select model", list_models())
    param_space = get_param_space(model)
elif problem_type == "Clustering":
    cluster = st.sidebar.selectbox("Select clustering algorithm", list_clusters())
    param_space = get_cluster_param_space(cluster)

st.sidebar.write("Parameters:")
params = {}
for p_name, spec in param_space.items():
    p_type = spec["type"]
    if p_type == "int":
        params[p_name] = st.sidebar.slider(
            p_name,
            min_value=int(spec["min"]), max_value=int(spec["max"]),
            step=int(spec.get("step", 1)),
            value=int(spec.get("default", spec["min"]))
        )
    elif p_type == "float":
        params[p_name] = st.sidebar.slider(
            p_name,
            min_value=float(spec["min"]), max_value=float(spec["max"]),
            step=float(spec.get("step", 0.01)),
            value=float(spec.get("default", spec["min"]))
        )
    elif p_type == "choice":
        params[p_name] = st.sidebar.selectbox(
            p_name,
            options=spec["choices"],
            index=spec["choices"].index(spec.get("default", spec["choices"][0]))
        )
    else:
        st.warning(f"Unsupported param type: {p_type} for {p_name}")

pca_dim = st.sidebar.selectbox("Choose PCA Dimension", [2, 3])
X_pca = pca_project(X, n_components=pca_dim)
ms = st.sidebar.slider("Marker Size", 1, 10, 1)
op = st.sidebar.slider("Opacity", 0.1, 1.0, 0.1)
plot_height = st.sidebar.slider("Plot height", 400, 1000, 100)

st.write(f"Shape of dataset: {X.shape}")
if problem_type == "Classification":
    st.write(f"Number of classes: {len(set(y))}")
    clf = build_model(model, params)
    result = train_and_eval(X, y, clf)
    st.write(f"Accuracy: {result['accuracy']:.2f}")
    y_pred = result['predictions']
else:
    st.write(f"Number of clusters: {len(set(y)) if y is not None else 'unknown'}")
    cluster_model = build_cluster(cluster, params)
    result = cluster_and_eval(X, y, cluster_model)
    st.write(f"Silhouette Score: {result['silhouette_score']:.2f}")
    st.write(f"Calinski-Harabasz Score: {result['calinski_harabasz_score']:.2f}")
    st.write(f"Davies-Bouldin Score: {result['davies_bouldin_score']:.2f}")
    y_pred = result['predictions']

if pca_dim == 2:
    fig = px.scatter(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    color=y.astype(str),
    labels={
        "x": "PC 1",
        "y": "PC 2",
        "color": "Klasse"
    },
    title="PCA-Projektion (2D) - " + dataset_name
)
else:
    fig = px.scatter_3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=X_pca[:, 2],
        color=y.astype(str),
        labels={
            "x": "PC 1",
            "y": "PC 2",
            "z": "PC 3",
            "color": "Klasse"
        },
        title="PCA-Projektion (3D) - " + dataset_name
    )
fig.update_traces(marker=dict(size=ms, opacity=op))
fig.update_layout(height=plot_height)
st.plotly_chart(fig, use_container_width=True)
