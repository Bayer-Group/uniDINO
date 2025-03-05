# %%
import scipy
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pycytominer import feature_select
from sklearn.metrics import accuracy_score

from source.eval import (
    nearest_neighbor_classifier_NSBW,
    get_feature_cols,
    create_gaussian_features,
)
from source import path

# %%
RANDOM_STATE = np.random.RandomState(322)

# %%
def get_actives(df, feature_type="cellprofiler", threshold=0.05):
    df = (
        df.query("source == 'Source 1'")
        .groupby(["perturbation_id"])
        .mean(numeric_only=True)
        .reset_index()
    )
    feature_columns, _ = get_feature_cols(df, feature_type=feature_type)
    features = df[feature_columns].values

    # 0.84 is 80th percentile of std. normal distribution
    actives = np.abs(features) >= 0.84

    percentage_actives = actives.mean(axis=1)
    df["percentage_actives"] = percentage_actives
    return df.query(f"percentage_actives > {threshold}")["perturbation_id"].values


# LOAD EMBEDDINGS
METRICS_ROOT = path.FIGURES / "jumpcp" / "metrics"

cell_profiler = pd.read_csv(path.JUMPCP.cell_profiler_mad_robustized)
cell_profiler = cell_profiler.loc[:, ~cell_profiler.columns.duplicated()].copy()
cell_profiler["group_broad"] = cell_profiler["group_broad"].fillna("No Annotation")
ACTIVE_COMPOUNDS = get_actives(cell_profiler)

cp_features = [
    c
    for c in cell_profiler.columns
    if c.startswith("Cell") or c.startswith("Cyto") or c.startswith("Nuc")
]
cp_meta_features = [c for c in cell_profiler.columns if c not in cp_features]


# %%
cell_profiler = feature_select(
    cell_profiler,
    features=cp_features,
    operation=[
        "variance_threshold",
        "correlation_threshold",
        "drop_na_columns",
        "blocklist",
    ],
)


def get_source(embeddings):
    return embeddings.merge(
        cell_profiler[["batch", "source"]].drop_duplicates(),
        on="batch",
        how="left",
    )


# %%
dino = pd.read_csv(path.JUMPCP.unidino_embeddings,index_col=False)
dino_transfer = pd.read_csv(path.JUMPCP.transfer_embeddings)
dino_transfer = get_source(dino_transfer)
dino_jumpcp_only = pd.read_csv(path.JUMPCP.jcp_only_embeddings)
dino_multi_channel = pd.read_csv(path.JUMPCP.jcp_multichannel_embeddings)
dino_multi_channel = get_source(dino_multi_channel)
microsnoop = pd.read_csv(path.JUMPCP.microsnoop_embeddings)
random_vit = pd.read_csv(path.JUMPCP.random_vit_embeddings)
gaussian = create_gaussian_features(dino, random_state=RANDOM_STATE)


# %%
def aggregate_profiles(df, consensus="well", annotation_variable="target"):
    if consensus not in ["well", "batch", "plate", "compound", "source"]:
        raise ValueError(
            f"consensus must be one of 'well', 'batch', 'plate', or 'compound'. "
            f"Got {consensus=}"
        )
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    grouping_features = [
        "perturbation_id",
        "batch",
        "well",
        "source",
        annotation_variable,
        "plate",
    ]
    if consensus == "plate":
        grouping_features.remove("well")
    elif consensus == "batch":
        grouping_features.remove("well")
        grouping_features.remove("plate")
    elif consensus == "source":
        grouping_features.remove("well")
        grouping_features.remove("plate")
        grouping_features.remove("batch")
    elif consensus == "compound":
        grouping_features.remove("well")
        grouping_features.remove("plate")
        grouping_features.remove("batch")
        grouping_features.remove("source")

    df = df.groupby(grouping_features).mean(numeric_only=True).reset_index()

    return df


def whitening_transform(X, lambda_, rotate=True):
    C = (1 / X.shape[0]) * np.dot(X.T, X)
    s, V = scipy.linalg.eigh(C)
    D = np.diag(1.0 / np.sqrt(s + lambda_))
    W = np.dot(V, D)
    if rotate:
        W = np.dot(W, V.T)
    return W


def whiten(X, mu, W):
    return np.dot(X - mu, W)


def TVN(embeddings, feature_type="standard"):
    # Source: https://github.com/broadinstitute/DeepProfilerExperiments/blob/master/bbbc021/profiling-evaluation.ipynb
    embeddings = embeddings.copy()
    reg_param = 1.0
    feature_ids, _ = get_feature_cols(embeddings, feature_type=feature_type)

    # Compute center of data around controls
    controls = embeddings["perturbation_id"] == "DMSO"
    mean_ctl = embeddings.loc[controls, feature_ids].mean()

    # Whitening transform on controls
    all_controls_matrix = embeddings.loc[controls, feature_ids] - mean_ctl
    W = whitening_transform(all_controls_matrix, reg_param, rotate=False)
    embeddings[feature_ids] = whiten(embeddings[feature_ids], mean_ctl, W)
    # Compute target matrix
    CF = embeddings.loc[embeddings["perturbation_id"] == "DMSO", feature_ids]
    C_target = (1 / CF.shape[0]) * np.dot(CF.T, CF)

    identity = reg_param * np.eye(C_target.shape[0])
    C_target = np.real(scipy.linalg.sqrtm(C_target + identity))

    aligned_feature_matrix = np.zeros_like(embeddings[feature_ids])
    plates = embeddings["batch"].unique()
    for p in plates:
        plate_data = embeddings[embeddings["batch"] == p]
        plate_index = plate_data.index.tolist()
        is_control = plate_data.perturbation_id == "DMSO"
        controls_index = plate_data[is_control].index

        controls_features = embeddings.iloc[controls_index][feature_ids]
        plate_features = embeddings.iloc[plate_index][feature_ids]

        C_source = (1 / controls_features.shape[0]) * np.dot(
            controls_features.T, controls_features
        )
        C_source = scipy.linalg.inv(
            np.real(scipy.linalg.sqrtm(C_source + np.eye(C_source.shape[0])))
        )

        X = np.dot(np.dot(plate_features, C_source), C_target)
        aligned_feature_matrix[plate_index, :] = X

    embeddings[feature_ids] = aligned_feature_matrix
    return embeddings


def preprocess_embedding(df, annotation="target"):
    # exclude drug classes with only one class member
    counts = df.value_counts(annotation).reset_index().rename(columns={0: "count"})
    counts = counts.query("count > 1")
    df = df[df[annotation].isin(counts[annotation])].reset_index(drop=True)
    return df


def get_metrics(
    embeddings,
    model_name="",
    feature_type="standard",
    use_TVN=False,
    moa_label="target",
):
    if use_TVN:
        embeddings = TVN(embeddings, feature_type=feature_type)

    embeddings = embeddings[
        embeddings["perturbation_id"].isin(ACTIVE_COMPOUNDS)
    ].reset_index(drop=True)

    batch_embeddings = aggregate_profiles(
        embeddings, consensus="batch", annotation_variable=moa_label
    )
    batch_embeddings = preprocess_embedding(
        batch_embeddings.query(f"{moa_label} != 'No Annotation'"),
        annotation=moa_label,
    )

    compound_embeddings = aggregate_profiles(
        embeddings, consensus="compound", annotation_variable=moa_label
    )
    compound_embeddings = preprocess_embedding(
        compound_embeddings.query(f"{moa_label} != 'No Annotation'"),
        annotation=moa_label,
    )

    print(f"{len(embeddings)=}, {len(batch_embeddings)=}, {len(compound_embeddings)=}")
    feature_columns, _ = get_feature_cols(embeddings, feature_type=feature_type)

    moa_pred_NSC_per_compound = nearest_neighbor_classifier_NSBW(
        compound_embeddings[feature_columns],
        compound_embeddings[moa_label],
        batches=compound_embeddings["perturbation_id"],
        mode="NSB",
    )

    moa_pred_NSC = nearest_neighbor_classifier_NSBW(
        batch_embeddings[feature_columns],
        batch_embeddings[moa_label],
        wells=batch_embeddings["perturbation_id"],
        mode="NSW",
    )

    moa_pred_NSCB = nearest_neighbor_classifier_NSBW(
        batch_embeddings[feature_columns],
        batch_embeddings[moa_label],
        batches=batch_embeddings["batch"],
        wells=batch_embeddings["perturbation_id"],
        mode="NSBW",
    )

    pert_pred_NSB_per_well = nearest_neighbor_classifier_NSBW(
        embeddings[feature_columns],
        embeddings["perturbation_id"],
        batches=embeddings["batch"],
        mode="NSB",
    )

    pert_pred_NSS_per_well = nearest_neighbor_classifier_NSBW(
        embeddings[feature_columns],
        embeddings["perturbation_id"],
        batches=embeddings["source"],
        mode="NSB",
    )

    moa_NSC_accuracy_compound = accuracy_score(
        moa_pred_NSC_per_compound, compound_embeddings[moa_label]
    )

    moa_NSC_accuracy = accuracy_score(moa_pred_NSC, batch_embeddings[moa_label])

    moa_NSCB_accuracy = accuracy_score(moa_pred_NSCB, batch_embeddings[moa_label])

    pert_NSB_accuracy = accuracy_score(
        pert_pred_NSB_per_well, embeddings["perturbation_id"]
    )

    pert_NSS_accuracy = accuracy_score(
        pert_pred_NSS_per_well, embeddings["perturbation_id"]
    )

    return pd.DataFrame(
        {
            "model": [model_name],
            f"NSC accuracy ({moa_label} / compound)": [moa_NSC_accuracy_compound],
            f"NSC accuracy ({moa_label} / batch)": [moa_NSC_accuracy],
            f"NSCB accuracy ({moa_label} / batch)": [moa_NSCB_accuracy],
            "NSB accuracy (compound / well)": [pert_NSB_accuracy],
            "NSS accuracy (compound / well)": [pert_NSS_accuracy],
        }
    ).round(3)


# %%
def get_metrics_wrapper(args):
    if len(args) == 4:
        return get_metrics(args[0], args[1], feature_type=args[2], use_TVN=args[3])
    else:
        return get_metrics(args[0], args[1], use_TVN=args[2])


for use_TVN in [False, True]:
    metrics_list = [
        (dino, "uniDINO", use_TVN),
        (microsnoop, "Microsnoop", use_TVN),
        (dino_transfer, "Transfer", use_TVN),
        (cell_profiler, "CellProfiler", "cellprofiler", use_TVN),
        (random_vit, "Random ViT", use_TVN),
        (gaussian, "Gaussian", use_TVN),
    ]

    metrics_results = Parallel(n_jobs=6)(
        delayed(get_metrics_wrapper)(args) for args in metrics_list
    )

    metrics = pd.concat(metrics_results).reset_index(drop=True)


    metrics.to_csv(
        METRICS_ROOT / f"jumpcp_metrics{'_TVN' if use_TVN else ''}.csv", index=False
    )
# %%
multi_channel_comparison = pd.concat(
    [
     get_metrics(dino_multi_channel, "Multi-Channel DINO", use_TVN=False),
     get_metrics(dino_jumpcp_only, "uniDINO JUMPCP only", use_TVN=False),
]).reset_index(drop=True)

multi_channel_comparison.to_csv(
    METRICS_ROOT / "multi_channel_metrics.csv", index=False
)

# %%
