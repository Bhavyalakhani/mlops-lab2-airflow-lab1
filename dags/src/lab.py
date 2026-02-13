import pandas as pd
from sklearn.preprocessing import StandardScaler
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
    """Load Wine dataset with basic stats and return as base64-encoded pickle."""
    wine_data = load_wine()
    features = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    features["target"] = wine_data.target
    
    logger.info("load_data: loaded %d samples with %d features", len(features), len(wine_data.feature_names))
    logger.info("load_data: feature stats - mean=%s, std=%s", 
                {col: round(features[col].mean(), 2) for col in wine_data.feature_names[:3]},
                {col: round(features[col].std(), 2) for col in wine_data.feature_names[:3]})
    
    if features.isnull().any().any():
        logger.warning("load_data: found null values")
    
    encode_data = pickle.dumps(features)
    encoded = base64.b64encode(encode_data).decode("ascii")
    logger.info("load_data: returning encoded data, size=%d bytes", len(encoded.encode()))
    return encoded

def data_preprocessing(data_b64: str):
    """Decode, scale with StandardScaler, persist scaler, return scaled data as base64."""
    logger.info("data_preprocessing: received encoded data, size=%d bytes", len(data_b64.encode()))
    payload_bytes = base64.b64decode(data_b64)
    df = pickle.loads(payload_bytes)
    
    df = df.dropna()
    feature_cols = sorted([c for c in df.columns if c != "target"])
    features_data = df[feature_cols]
    
    logger.info("data_preprocessing: input shape=%s, features=%d", df.shape, len(feature_cols))
    
    # Apply StandardScaler for z-score normalization
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_data)
    
    logger.info("data_preprocessing: scaled with StandardScaler, output shape=%s", scaled_data.shape)
    
    # Save scaler for prediction task
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(model_dir, exist_ok=True)
    scaler_file = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("data_preprocessing: scaler saved to %s", scaler_file)
    
    encoded = base64.b64encode(pickle.dumps(scaled_data)).decode("ascii")
    logger.info("data_preprocessing: payload encoded, size=%d bytes", len(encoded))
    return encoded


def build_save_model(data_b64: str, filename: str):
    """Train KMeans across k range, detect elbow, save final model, return loss history."""
    logger.info("build_save_model: received encoded data, size=%d bytes", len(data_b64.encode()))
    payload_bytes = base64.b64decode(data_b64)
    features = pickle.loads(payload_bytes)
    
    logger.info("build_save_model: received features shape=%s", features.shape)
    logger.info("build_save_model: output model=%s", filename)
    
    k_search = range(1, 50)
    losses = []
    
    logger.info("build_save_model: training KMeans for %d k values", len(list(k_search)))
    
    for k in k_search:
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=42)
        km.fit(features)
        losses.append(float(km.inertia_))
        if k % 10 == 1:
            logger.info("build_save_model: k=%d, loss=%.4f", k, losses[-1])
    
    logger.info("build_save_model: training complete, losses=%s", losses[:5])
    
    # Detect optimal k with elbow method
    try:
        knee = KneeLocator(list(k_search), losses, curve="convex", direction="decreasing")
        best_k = int(knee.elbow) if knee.elbow is not None else 3
        logger.info("build_save_model: elbow at k=%d", best_k)
    except Exception as e:
        best_k = 3
        logger.warning("build_save_model: elbow detection failed, using k=%d", best_k)
    
    # Train final model at best k
    final_km = KMeans(n_clusters=best_k, init="k-means++", n_init=20, random_state=42)
    final_km.fit(features)
    
    # Save model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    with open(model_path, "wb") as f:
        pickle.dump(final_km, f)
    
    logger.info("build_save_model: model saved to %s (k=%d)", model_path, best_k)
    logger.info("build_save_model: returning losses, size=%d bytes", len(pickle.dumps(losses)))
    return losses


def load_model_elbow(filename: str, losses: list):
    """Load model, recompute elbow, scale and predict on Wine data."""
    logger.info("load_model_elbow: received losses, size=%d bytes", len(pickle.dumps(losses)))
    # Load trained model
    model_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    with open(model_path, "rb") as f:
        trained_model = pickle.load(f)
    
    logger.info("load_model_elbow: loaded model from %s", model_path)
    
    # Recompute elbow analysis
    k_range = list(range(1, len(losses) + 1))
    try:
        knee = KneeLocator(k_range, losses, curve="convex", direction="decreasing")
        logger.info("load_model_elbow: elbow suggests k=%s", knee.elbow)
    except Exception as e:
        logger.warning("load_model_elbow: elbow detection failed (%s)", e)
    
    # Load Wine dataset and prepare for prediction
    wine = load_wine()
    raw_data = wine.data
    
    logger.info("load_model_elbow: loaded Wine data, shape=%s", raw_data.shape)
    
    # Load and apply saved scaler
    scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "scaler.pkl")
    
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        try:
            raw_df = pd.DataFrame(raw_data, columns=getattr(scaler, "feature_names_in_", None))
            scaled_data = scaler.transform(raw_df)
        except Exception:
            scaled_data = scaler.transform(raw_data)
        
        logger.info("load_model_elbow: scaled data with saved scaler, shape=%s", scaled_data.shape)
    else:
        scaled_data = raw_data
        logger.warning("load_model_elbow: scaler not found, using raw data")
    
    # Predict
    pred = trained_model.predict(scaled_data)[0]
    logger.info("load_model_elbow: prediction made, cluster=%s, data_size=%s", pred, scaled_data.size)
    result = int(pred)
    logger.info("response size=%d bytes", len(pickle.dumps(result)))
    return result

# created this while testing   
# if __name__ == '__main__':
#     data = load_data()
#     preprocessed_data = data_preprocessing(data)
#     sse = build_save_model(preprocessed_data, 'wine_model.pkl')
#     response = load_model_elbow("wine_model.pkl", sse)
#     print(response)
