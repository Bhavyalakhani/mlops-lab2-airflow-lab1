import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64
from sklearn.datasets import load_wine
import logging

# module logger
logger = logging.getLogger(__name__)

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    # Load the Wine dataset from scikit-learn and return as pickled DataFrame
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    logger.info("load_data: loaded Wine dataset with %d samples and %d features", df.shape[0], df.shape[1]-1)
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    # select all feature columns (exclude target)
    feature_cols = [c for c in df.columns if c != "target"]
    clustering_data = df[feature_cols]

    logger.info("data_preprocessing: input dataframe shape=%s, feature_count=%d", df.shape, len(feature_cols))
    logger.info("data_preprocessing: feature columns sample=%s", feature_cols[:5])

    # scale features (MinMaxScaler for a simple bounded range)
    scaler = MinMaxScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)

    logger.info("data_preprocessing: scaled data shape=%s (rows,cols)", getattr(clustering_data_scaled, 'shape', None))

    # save scaler for later use by prediction step
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("data_preprocessing: scaler saved to %s", scaler_path)

    # bytes -> base64 string for XCom
    clustering_serialized_data = pickle.dumps(clustering_data_scaled)
    encoded = base64.b64encode(clustering_serialized_data).decode("ascii")
    logger.info("data_preprocessing: returning base64 payload size=%d bytes", len(encoded))
    return encoded


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model on the preprocessed data and saves it.
    Returns the SSE list (JSON-serializable).
    """

    data_bytes = base64.b64decode(data_b64)
    X = pickle.loads(data_bytes)

    logger.info("build_save_model: received data for training, shape=%s", getattr(X, 'shape', None))
    logger.info("build_save_model: target filename=%s", filename)

    k_range = range(1, 50)
    kmeans_kwargs = {"init": "k-means++", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        sse.append(float(kmeans.inertia_))

    logger.info("build_save_model: computed SSE for %d k values (sample first 5)=%s", len(sse), sse[:5])

    try:
        kl = KneeLocator(list(k_range), sse, curve="convex", direction="decreasing")
        chosen_k = int(kl.elbow) if kl.elbow is not None else 3
        logger.info("build_save_model: KneeLocator found elbow at k=%s", chosen_k)
    except Exception as e:
        chosen_k = 3
        logger.warning("build_save_model: KneeLocator failed (%s) — defaulting chosen_k=%s", e, chosen_k)

    final_kmeans = KMeans(n_clusters=chosen_k, **kmeans_kwargs)
    final_kmeans.fit(X)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(final_kmeans, f)
    logger.info("build_save_model: final KMeans (k=%s) saved to %s", chosen_k, output_path)

    # return SSE list (JSON-serializable)
    return sse


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report k.
    """
    # load the saved model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # elbow for information/logging — build x range from sse length
    x_range = list(range(1, len(sse) + 1))
    kl = KneeLocator(x_range, sse, curve="convex", direction="decreasing")
    logger.info("load_model_elbow: suggested elbow k=%s", kl.elbow)

    # load the wine dataset and prepare features for prediction
    wine = load_wine()
    X = wine.data

    scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = pickle.load(open(scaler_path, "rb"))
        # try to preserve feature-name alignment to avoid warnings
        try:
            X_df = pd.DataFrame(X, columns=getattr(scaler, "feature_names_in_", None))
            X_scaled = scaler.transform(X_df)
        except Exception:
            X_scaled = scaler.transform(X)
        logger.info("load_model_elbow: transformed input data using saved scaler, shape=%s", getattr(X_scaled, 'shape', None))
    else:
        X_scaled = X
        logger.info("load_model_elbow: no scaler found, using raw data shape=%s", getattr(X_scaled, 'shape', None))

    pred = loaded_model.predict(X_scaled)[0]

    # ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred

# created this while testing   
# if __name__ == '__main__':
#     data = load_data()
#     preprocessed_data = data_preprocessing(data)
#     sse = build_save_model(preprocessed_data, 'wine_model.pkl')
#     response = load_model_elbow("wine_model.pkl", sse)
#     print(response)




