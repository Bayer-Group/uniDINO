from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from source.utils import assert_

def create_gaussian_features(embeddings, random_state=None):
    if random_state is None:
        random_state = np.random.default_rng()

    gaussian_embeddings = embeddings.copy()
    n_samples = len(embeddings)
    feature_columns = [c for c in embeddings.columns if c.startswith("emb")]
    for feature_column in feature_columns:
        gaussian_embeddings[feature_column] = random_state.standard_normal(
            size=n_samples
        )
    return gaussian_embeddings

def get_feature_cols(df, feature_type="standard"):
    if feature_type == "standard":
        feature_columns = [c for c in df.columns if c.startswith("emb")]
    elif feature_type == "cellprofiler":
        feature_columns = [
            c
            for c in df.columns
            if c.startswith("Cell") or c.startswith("Cyto") or c.startswith("Nuc")
        ]
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    meta_columns = [c for c in df.columns if c not in feature_columns]
    return feature_columns, meta_columns

def pairwise_distances_parallel(
    X, metric="euclidean", n_jobs=None, min_samples=15000, chunk_size=5000
):
    """Splits computation of distance matrix into several chuncks for very large only"""

    if X.shape[0] >= min_samples:
        max_i = int(np.ceil(X.shape[0] / chunk_size))
        chunk_idxs = [
            [i * chunk_size, min(X.shape[0], (i + 1) * chunk_size)]
            for i in range(max_i)
        ]
        dist_mat = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(pairwise_distances)(X[idx[0] : idx[1], :], X, metric)
            for idx in chunk_idxs
        )
        dist_mat = np.concatenate(dist_mat)
    else:
        dist_mat = pairwise_distances(X, metric=metric)

    return dist_mat

def nearest_neighbor_classifier_NSBW(
    X, y, mode="NN", batches=[], wells=[], metric="cosine"
):
    """
    Get the label of its nearest neighbor for each sample in X but, optionally, excluding all samples which belong to the same batch or the same well
    :param X: feature matrix (shape = samples, features)
    :param y: labels vector (len = samples )
    :param mode: Type of classifier taks
        - NN: simple nearest neighbor conisdering all samples in the data
        - NSB: not same bacth
        - NSW: not same well
        - NSBW: not same batch or well
    :param batches: vector with batch assignments (len = samples )
    :param wells: vector with well assignments (len = samples )
    """
    assert_(mode in ["NN", "NSB", "NSW", "NSBW"], "unknown mode")

    dist_mat = pairwise_distances_parallel(X, metric=metric)
    max_dist = dist_mat.max()
    np.fill_diagonal(dist_mat, max_dist)

    # For each compound, penalize compounds from the same batch and/or well
    # by assigning them the maximum distance
    if mode != "NN":
        for well_idx in range(X.shape[0]):
            if "B" in mode:
                if not batches:
                    raise ValueError("Batches must be provided for NSB* mode")
                same_batch_idx = np.logical_and(
                    batches == batches[well_idx], y == y[well_idx]
                )
                dist_mat[well_idx, same_batch_idx] = max_dist
            if "W" in mode:
                if not wells:
                    raise ValueError("Wells must be provided for NS*W mode")
                same_well_idx = wells == wells[well_idx]
                dist_mat[well_idx, same_well_idx] = max_dist

    knn_idxs = np.argmin(dist_mat, axis=1)
    y_pred = y[knn_idxs]

    return y_pred

def pca_reduce(embeddings, n_components=384, feature_type="standard"):
    feature_columns, meta_columns = get_feature_cols(embeddings, feature_type=feature_type)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components))  # Adjust n_components as needed
    ])

    pca_projection = pipeline.fit_transform(embeddings[feature_columns].values)
    pca_df = pd.DataFrame(columns=[f"emb{i:03}" for i in range(n_components)], data=pca_projection)
    return pd.concat((embeddings[meta_columns], pca_df), axis=1)
