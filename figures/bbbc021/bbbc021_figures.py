# %%
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP

from source.eval import (
    nearest_neighbor_classifier_NSBW,
    get_feature_cols,
    create_gaussian_features,
    )
from source import path

from pycytominer import feature_select

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = np.random.RandomState(322)


# LOAD EMBEDDINGS

MOA_PALETTE = {
    "Actin disruptors": "#CD2626",
    "DNA replication": "#7D26CD",
    "DNA damage": "#9F79EE",
    "Microtubule stabilizers": "#27408B",
    "Microtubule destabilizers": "#FFD700",
    "Eg5 inhibitors": "#4F94CD",
    "Epithelial": "#B0E2FF",
    "Aurora kinase inhibitors": "#FF6347",
    "Kinase inhibitors": "#889898",
    "Protein degradation": "#00A410",
    "Protein synthesis": "#EEAEEE",
    "Cholesterol-lowering": "#7CCD7C",
    "DMSO": "#DDD8D8",
}

FIGURES_ROOT =  path.FIGURES / "bbbc021"

dino = pd.read_csv(path.BBBC021.unidino_embeddings, index_col=False)
dino_transfer = pd.read_csv(path.BBBC021.transfer_embeddings)
microsnoop = pd.read_csv(path.BBBC021.microsnoop_embeddings).drop(columns=["ImageNumber"])
random_vit = pd.read_csv(path.BBBC021.random_vit_embeddings)
gaussian = create_gaussian_features(dino, random_state=RANDOM_STATE)

dino_jcp_only = pd.read_csv(path.BBBC021.jcp_only_embeddings)
dino_jcp_hpa = pd.read_csv(path.BBBC021.jcp_hpa_embeddings)
dino_jcp_hpa_b21 = pd.read_csv(path.BBBC021.jcp_hpa_b21_embeddings)


# The plate normalization here was done only based on the wells with moas!
cell_profiler_raw = (
    pd.read_csv(path.BBBC021.cell_profiler_raw)
    .rename(
        columns={
            "Treatment_Name": "compound",
            "Treatment_Concentration": "concentration",
            "Treatment_Label__Moa": "moa",
            "Plate_ID": "plate",
            "Batch_ID": "TableNumber",
            "Well_ID": "well",
        }
    )
    .drop(
        columns=[
            "Treatment_Type",
            "Treatment_Smiles",
            "CellType",
            "ControlType",
            "Replicate",
        ]
    )
)
cell_profiler = (
    pd.read_csv(path.BBBC021.cell_profiler_mad_robustized)
    .rename(
        columns={
            "Treatment_Name": "compound",
            "Treatment_Concentration": "concentration",
            "Treatment_Label__Moa": "moa",
            "Plate_ID": "plate",
            "Batch_ID": "TableNumber",
            "Well_ID": "well",
        }
    )
    .drop(
        columns=[
            "Treatment_Type",
            "Treatment_Smiles",
            "CellType",
            "ControlType",
            "Replicate",
        ]
    )
    .iloc[:, 1:]
)

cell_profiler = cell_profiler.loc[:, ~cell_profiler.columns.duplicated()].copy()

cp_features = [
    c
    for c in cell_profiler.columns
    if c.startswith("Cell") or c.startswith("Cyto") or c.startswith("Nuc")
]
cp_meta_features = [c for c in cell_profiler.columns if c not in cp_features]
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


# %%
# FUNCTIONS
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
    controls = embeddings["compound"] == "DMSO"
    mean_ctl = embeddings.loc[controls, feature_ids].mean()

    # Whitening transform on controls
    all_controls_matrix = embeddings.loc[controls, feature_ids] - mean_ctl
    W = whitening_transform(all_controls_matrix, reg_param, rotate=False)
    embeddings[feature_ids] = whiten(embeddings[feature_ids], mean_ctl, W)
    # Compute target matrix
    CF = embeddings.loc[embeddings["compound"] == "DMSO", feature_ids]
    C_target = (1 / CF.shape[0]) * np.dot(CF.T, CF)

    identity = reg_param * np.eye(C_target.shape[0])
    C_target = np.real(scipy.linalg.sqrtm(C_target + identity))

    aligned_feature_matrix = np.zeros_like(embeddings[feature_ids])
    plates = embeddings["batch"].unique()
    for p in plates:
        plate_data = embeddings[embeddings["batch"] == p]
        plate_index = plate_data.index.tolist()
        is_control = plate_data.compound == "DMSO"
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


def aggregate_profiles(df, consensus="well"):
    if consensus not in ["well", "batch", "plate", "treatment"]:
        raise ValueError(
            f"consensus must be one of 'batch', 'plate', or 'treatment'. "
            f"Got {consensus=}"
        )
    df["condition_id"] = df["compound"] + df["concentration"].astype(str)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    grouping_features = ["condition_id", "compound", "batch", "well", "moa", "plate"]
    if consensus == "plate":
        grouping_features.remove("well")
    elif consensus == "batch":
        grouping_features.remove("well")
        grouping_features.remove("plate")
    elif consensus == "treatment":
        grouping_features.remove("well")
        grouping_features.remove("plate")
        # grouping_features.remove("batch") # The treatments are in different batches

    df = df.groupby(grouping_features).mean(numeric_only=True).reset_index()

    return df


def get_umap_features(df, consensus=None, feature_type="standard"):
    df = aggregate_profiles(df, consensus=consensus)
    feature_columns, meta_columns = get_feature_cols(df, feature_type=feature_type)
    umap = UMAP(densmap=True, random_state=RANDOM_STATE)
    umap_projection = umap.fit_transform(df[feature_columns].values)
    umap_df = pd.DataFrame(columns=["UMAP 1", "UMAP 2"], data=umap_projection)
    return pd.concat((df[meta_columns], umap_df), axis=1)


def plot_umap(embeddings, label, model_name, feature_type="standard", fontsize=17):
    umap_df = get_umap_features(
        embeddings.query("moa != 'No MoA'"),
        consensus="treatment",
        feature_type=feature_type,
    )
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=umap_df,
        x="UMAP 1",
        y="UMAP 2",
        hue=label,
        hue_order=MOA_PALETTE.keys(),
        palette=MOA_PALETTE.values(),
        s=120,
    )
    plt.legend(
        bbox_to_anchor=(0.5, -0.15),
        loc="upper center",
        borderaxespad=0,
        fontsize=fontsize,
        frameon=False,
        ncol=2,
    )
    sns.despine(top=True, right=True)

    plt.xlabel("UMAP 1", fontsize=fontsize)
    plt.ylabel("UMAP 2", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.savefig(FIGURES_ROOT / "images" / f"umap_{model_name}.pdf", bbox_inches="tight")
    plt.clf()


def get_clustermap_features(df, feature_type="standard"):
    moa_treatment = (
        df.query("moa != 'No MoA'")
        .groupby(["compound", "moa"])
        .mean(numeric_only=True)
        .reset_index()
    )
    feature_cols, _ = get_feature_cols(moa_treatment, feature_type=feature_type)
    moa_corr = moa_treatment[feature_cols].transpose().corr("pearson")
    moa_corr.columns = moa_corr.index = moa_treatment["moa"]

    return moa_corr


def plot_clustermap(clustermap_df, model_name="", feature_type="standard"):
    clustermap_df = get_clustermap_features(clustermap_df, feature_type=feature_type)
    sns.set_theme(font_scale=1.6)
    clustermap = sns.clustermap(
        clustermap_df,
        cmap="bwr",
        figsize=(15, 15),
        method="average",
        dendrogram_ratio=0.05,
        cbar_kws={"pad": 20, "orientation": "horizontal"},
        cbar_pos=(0.04, 1.01, 0.08, 0.01),
        vmin=-1,
        vmax=1,
    )

    clustermap.ax_heatmap.set_xticklabels([])
    clustermap.ax_heatmap.set_xticks([])
    clustermap.ax_heatmap.set_ylabel("")
    clustermap.ax_heatmap.set_xlabel("")

    plt.savefig(
        FIGURES_ROOT / "images" / f"treatment_heatmap_{model_name}.pdf",
        bbox_inches="tight",
    )
    plt.clf()


def get_moa_metrics(embeddings, model_name, feature_type="standard", tvn=False):
    embeddings = embeddings.query("moa != 'No MoA'").reset_index(drop=True)
    if tvn:
        embeddings = TVN(embeddings, feature_type=feature_type)
    embeddings = embeddings[embeddings["moa"].notna()]
    embeddings = embeddings.query(
        (
            "~(compound == 'DMSO' | (compound == 'taxol' "
            "& ~(plate.str.contains('Week1_') & well in ['D03', 'D04', 'D05'])))"
        )
    ).reset_index(drop=True)
    # Only in one batch
    nscb_embeddings = embeddings[
        ~embeddings["moa"].isin(["Cholesterol-lowering", "Kinase inhibitors"])
    ].reset_index(drop=True)
    treatment_embeddings = aggregate_profiles(embeddings, consensus="treatment")

    print(f"{len(embeddings)=}, {len(treatment_embeddings)=}, {len(nscb_embeddings)=}")

    feature_columns, _ = get_feature_cols(embeddings, feature_type=feature_type)
    y_pred_NSC_per_treatment = nearest_neighbor_classifier_NSBW(
        treatment_embeddings[feature_columns],
        treatment_embeddings["moa"],
        batches=treatment_embeddings["compound"],
        mode="NSB",
    )

    y_pred_NSC_per_well = nearest_neighbor_classifier_NSBW(
        embeddings[feature_columns],
        embeddings["moa"],
        wells=embeddings["compound"],
        mode="NSW",
    )

    y_pred_NSCB_per_well = nearest_neighbor_classifier_NSBW(
        nscb_embeddings[feature_columns],
        nscb_embeddings["moa"],
        batches=nscb_embeddings["batch"],
        wells=nscb_embeddings["compound"],
        mode="NSBW",
    )

    y_pred_NSPW_per_well = nearest_neighbor_classifier_NSBW(
        embeddings[feature_columns],
        embeddings["compound"],
        batches=embeddings["plate"],
        wells=embeddings["well"],
        mode="NSBW",
    )

    NSC_accuracy_per_treatment = accuracy_score(
        y_pred_NSC_per_treatment, treatment_embeddings["moa"]
    )

    NSC_accuracy_per_well = accuracy_score(y_pred_NSC_per_well, embeddings["moa"])
    NSCB_accuracy_per_well = accuracy_score(
        y_pred_NSCB_per_well, nscb_embeddings["moa"]
    )

    NSPW_accuracy_per_well = accuracy_score(
        y_pred_NSPW_per_well, embeddings["compound"]
    )

    return pd.DataFrame(
        {
            "model": [model_name],
            "NSC accuracy (treatment)": [NSC_accuracy_per_treatment],
            "NSC accuracy (well)": [NSC_accuracy_per_well],
            "NSCB accuracy (well)": [NSCB_accuracy_per_well],
            "NSPW accuracy (well)": [NSPW_accuracy_per_well],
        }
    ).round(3)


def get_feature_importance(df, label):
    feature_columns = [c for c in df.columns if c.startswith("emb")]
    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE, n_estimators=100, class_weight="balanced"
    )
    classifier.fit(df[feature_columns], df[label])
    feature_importances = pd.DataFrame(
        {"feature": feature_columns, "importance": classifier.feature_importances_}
    ).sort_values(ascending=False, by="importance", ignore_index=True)

    feature_importances["channel"] = feature_importances["feature"].apply(
        get_feature_channel
    )
    return feature_importances


def top_feature_importance_per_moa(df, n_features=100):
    moa_feature_importance = []
    for moa in sorted(df["moa"].unique()):
        df[moa] = df["moa"] == moa
        feature_importance = get_feature_importance(df, moa)
        feature_importance = feature_importance.head(n_features).value_counts("channel")
        feature_importance = pd.DataFrame(feature_importance).T
        feature_importance["MoA"] = moa
        moa_feature_importance.append(feature_importance)
    moa_feature_importance = pd.concat(moa_feature_importance)
    moa_feature_importance.columns.names = [""]
    return moa_feature_importance.reset_index(drop=True)


def get_feature_channel(entry):
    feature_number = int(entry.removeprefix("emb"))
    if feature_number < 384:
        return "DNA"
    elif feature_number >= 768:
        return "Actin"
    else:
        return "Tubulin"


# %%
# METRICS
moa_metrics = pd.concat(
    [
        get_moa_metrics(dino, "uniDINO"),
        get_moa_metrics(microsnoop, "Microsnoop"),
        get_moa_metrics(
            dino_transfer.rename(columns={"TableNumber": "batch"}), "Transfer"
        ),
        get_moa_metrics(
            cell_profiler.rename(columns={"TableNumber": "batch"}),
            "CellProfiler",
            feature_type="cellprofiler",
        ),
        get_moa_metrics(
            random_vit.rename(columns={"TableNumber": "batch"}), "Random ViT"
        ),
        get_moa_metrics(gaussian, "Gaussian"),
    ]
).reset_index(drop=True)

# %%
moa_metrics.to_csv(FIGURES_ROOT / "moa_metrics.csv", index=False)

# %%
moa_metrics_TVN = pd.concat(
    [
        get_moa_metrics(dino, "uniDINO", tvn=True),
        get_moa_metrics(microsnoop, "Microsnoop", tvn=True),
        get_moa_metrics(
            dino_transfer.rename(columns={"TableNumber": "batch"}),
            "Transfer",
            tvn=True,
        ),
        get_moa_metrics(
            cell_profiler_raw.rename(columns={"TableNumber": "batch"}),
            "CellProfiler",
            feature_type="cellprofiler",
            tvn=True,
        ),
        get_moa_metrics(
            random_vit.rename(columns={"TableNumber": "batch"}), "Random ViT", tvn=True
        ),
        get_moa_metrics(gaussian, "Gaussian", tvn=True),
    ]
).reset_index(drop=True)
#%%
moa_metrics_TVN.to_csv(FIGURES_ROOT / "moa_metrics_TVN.csv", index=False)

#%%
scaling_metrics = pd.concat(
    [
        get_moa_metrics(dino_jcp_only, "JUMP-CP only"),
        get_moa_metrics(dino_jcp_hpa, "JUMP-CP + HPA"),
        get_moa_metrics(dino_jcp_hpa_b21, "JUMP-CP + HPA + BBBC021"),
        get_moa_metrics(dino, "uniDINO"),
    ]
).reset_index(drop=True)
#%%
scaling_metrics.to_csv(FIGURES_ROOT / "scaling_metrics.csv", index=False)

# %%
# UMAP
plot_umap(dino, "moa", "DINO")
plot_umap(microsnoop, "moa", "Microsnoop")
plot_umap(dino_transfer.rename(columns={"TableNumber": "batch"}), "moa", "Transfer")
plot_umap(
    cell_profiler.rename(columns={"TableNumber": "batch"}),
    "moa",
    "CellProfiler",
    feature_type="cellprofiler",
)
plot_umap(random_vit.rename(columns={"TableNumber": "batch"}), "moa", "Random ViT")

# %%
plot_umap(
    dino.query("moa in ('Microtubule destabilizers', 'DNA replication')").reset_index(
        drop=True
    ),
    "moa",
    "DINO_abstract",
)

# CLUSTERMAP
# %%
plot_clustermap(dino, model_name="DINO", feature_type="standard")
plot_clustermap(microsnoop, model_name="Microsnoop", feature_type="standard")
plot_clustermap(
    dino_transfer.rename(columns={"TableNumber": "batch"}),
    model_name="Transfer",
    feature_type="standard",
)
plot_clustermap(
    cell_profiler.rename(columns={"TableNumber": "batch"}),
    model_name="CellProfiler",
    feature_type="cellprofiler",
)
plot_clustermap(
    random_vit.rename(columns={"TableNumber": "batch"}),
    model_name="Random ViT",
    feature_type="standard",
)

# FEATURE IMPORTANCE
# %%
abstract = False
dino_moa = aggregate_profiles(dino, consensus="plate").query("moa != 'No MoA'")
moa_feature_importance = top_feature_importance_per_moa(dino_moa, n_features=50)
if abstract:
    moa_feature_importance = moa_feature_importance.query(
        "MoA in ('Microtubule destabilizers', 'DNA replication')"
    ).reset_index(drop=True)
feature_importance_fontsize = 22

sns.set_palette("colorblind")
moa_feature_importance.fillna(0).sort_values(["Actin", "Tubulin"]).set_index(
    "MoA"
).plot(kind="bar", stacked=True, figsize=(15, 10))
sns.despine(top=True, right=True)
plt.legend(
    bbox_to_anchor=(1.2, 0.5),
    loc="upper right",
    borderaxespad=0.0,
    frameon=False,
    fontsize=feature_importance_fontsize,
)
plt.xticks(
    rotation=45,
    ha="right",
    rotation_mode="anchor",
    fontsize=feature_importance_fontsize,
)
plt.yticks(fontsize=feature_importance_fontsize)
plt.xlabel(
    "Mechanism of Action (MoA)", fontsize=feature_importance_fontsize
)  # Increase x-label font size
plt.ylabel("Number of Features", fontsize=feature_importance_fontsize)
plt.tight_layout(rect=(0., 0., 1., 1.))

plot_suffix = "_abstract" if abstract else ""
plt.savefig(FIGURES_ROOT / "images" / f"feature_importance{plot_suffix}.pdf")
