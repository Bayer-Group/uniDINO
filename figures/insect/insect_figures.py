# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from umap import UMAP

from source.eval import (
    nearest_neighbor_classifier_NSBW,
    get_feature_cols,
    create_gaussian_features,
)
from source import path

import pycytominer

RANDOM_STATE = np.random.RandomState(398)
MOA_PALETTE = {
    "#8B8989": "AChE & BuChE inhibitor",
    "#7D26CD": "Acetyl-CoA carboxylase inhibitor",
    "#C1D2FF": "Actin polymerization inhibitor",
    "#EEAEEE": "Apoptosis inducer",
    "#CDC0B0": "Mitochondrial ATPase inhibitor",
    "#FFB90F": "Mitochondrial complex I inhibitor",
    "#27408B": "Mitochondrial complex II inhibitor",
    "#2E8B57": "Mitochondrial complex III inhibitor",
    "#FFD700": "Mitochondrial uncoupler",
    "#FF1188": "Mitosis inhibitor / Tubulin disruptor",
    "#EEE8E8": "Negative control",
    "#FF7F50": "Neurotoxin",
    "#000000": "Signal transduction inhibitor",
    "#FFEC8B": "V-ATPase inhibitor",
}

# %%
FOLDER_ROOT = path.FIGURES / "insect"
IMAGES_ROOT = FOLDER_ROOT / "images"

annotations = pd.read_excel(
    FOLDER_ROOT / "cluster_biological_annot.xlsx"
)
selected_concentration_plates_1_to_3 = pd.read_excel(
    FOLDER_ROOT / "cluster_selected_conc_P1-P3.xlsx"
)
selected_concentration_plates_4_to_6 = pd.read_excel(
    FOLDER_ROOT / "cluster_selected_conc_P4-P6.xlsx"
)

# %%
perturbation_id_map = {
    "Abamectin": "Abamectin B1",
    "Atpenin": "Atpenin A5",
    "Bafilomycin": "Bafilomycin A1",
    "Cytocalasin D": "Cytochalasin D",
    "Fenbutationoxid": "Fenbutatinoxide",
    "Latruculin B": "Latrunculin B",
    "Saccharose": "Sucrose",
    "Taxol": "Paclitaxel",
}

dino = (
    pd.read_csv(path.Insect.unidino_embeddings)
    .replace(perturbation_id_map)
    .merge(annotations)
)

dino_jumpcp_only = (
    pd.read_csv(path.Insect.jcp_only_embeddings)
    .replace(perturbation_id_map)
    .merge(annotations)
)

dino_jumpcp_hpa = (
    pd.read_csv(path.Insect.jcp_hpa_embeddings)
    .replace(perturbation_id_map)
    .merge(annotations)
)

dino_jumpcp_hpa_b21 = (
    pd.read_csv(path.Insect.jcp_hpa_b21_embeddings)
    .replace(perturbation_id_map)
    .merge(annotations)
)

microsnoop = (
    pd.read_csv(path.Insect.microsnoop_embeddings)
    .replace(perturbation_id_map)
    .merge(annotations)
)

transfer = (
    pd.read_csv(path.Insect.transfer_embeddings)
    .replace(perturbation_id_map)
    .merge(annotations)
)

random = (
    pd.read_csv(path.Insect.randomvit_embeddings)
    .replace(perturbation_id_map)
    .merge(annotations)
)

cell_profiler = (
    pd.read_csv(path.Insect.cellprofiler)
    .replace(perturbation_id_map)
    .merge(annotations)
    .query("batch =='Batch3'")
    .reset_index(drop=True)
    .astype({"concentration": "float64"})
)

feature_select_operations = [
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
]
cell_profiler = pycytominer.feature_select(
    cell_profiler, corr_threshold=0.9, operation=feature_select_operations
)


# %%
def get_consensus_embeddings(embeddings):
    embeddings["treatment_id"] = (
        embeddings["perturbation_id"] + "_" + embeddings["concentration"].astype(str)
    )
    embeddings = embeddings.drop(columns=["well", "cell_count", "batch"])
    consensus_embeddings = (
        embeddings.groupby(["treatment_id", "annotation", "perturbation_id"])
        .mean(numeric_only=True)
        .reset_index()
    )
    return consensus_embeddings


def get_correlation_df(consensus_embeddings):
    correlation_df = (
        consensus_embeddings.drop(columns=["treatment_id", "annotation"])
        .transpose()
        .corr("pearson")
    )
    correlation_df.columns = consensus_embeddings["treatment_id"]
    correlation_df.index = consensus_embeddings["treatment_id"]
    return correlation_df


def get_umap_features(embeddings, n_neighbors=15, min_dist=0.8, metric="euclidean"):
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    features = embeddings.filter(regex=r"^(emb)")
    h = reducer.fit_transform(features.values)
    umap_df = pd.DataFrame({"umap_x": h[:, 0], "umap_y": h[:, 1]})
    meta_columns = [c for c in embeddings.columns if c not in features.columns]
    return pd.concat([embeddings[meta_columns], umap_df], axis=1)


def batch_aggregate(data_df):
    feature_cols, _ = get_feature_cols(data_df)
    batch_agg_df = (
        data_df.groupby(["batch", "perturbation_id"], dropna=False)
        .agg(
            {
                col: "mean" if col in feature_cols + ["cell_count"] else "first"
                for col in data_df.columns
            }
        )
        .reset_index(drop=True)
    )
    return batch_agg_df


def get_metrics(
    well_embeddings, model_name, selected_wells=True, feature_type="standard"
):
    if selected_wells:
        well_embeddings = get_selected_wells(well_embeddings)
    well_embeddings["annotation"] = (
        well_embeddings["annotation"]
        .str.strip()
        .replace(
            {
                "Mitochondrial complex I inhibitor": "Mitochondrial complex inhibitor",
                "Mitochondrial complex II inhibitor": "Mitochondrial complex inhibitor",
                "Mitochondrial complex III inhibitor": "Mitochondrial complex inhibitor",
            }
        )
    )
    included_annotations = [
        "Actin polymerization inhibitor",
        "Mitochondrial ATPase inhibitor",
        "V-ATPase inhibitor",
        "Mitosis inhibitor / Tubulin disruptor",
        "Mitochondrial complex inhibitor",
    ]

    well_embeddings = well_embeddings.query(
        f"annotation in {included_annotations}"
    ).reset_index(drop=True)
    well_embeddings["treatment_id"] = (
        well_embeddings["perturbation_id"]
        + "_"
        + well_embeddings["concentration"].astype(str)
    )

    treatment_embeddings = (
        well_embeddings.groupby(["treatment_id", "perturbation_id", "annotation"])
        .mean(numeric_only=True)
        .reset_index()
    )

    print(
        f"Number of treatments embeddings: {len(treatment_embeddings)}\n"
        f"Number of well embeddings: {len(well_embeddings)}"
    )

    feature_columns, _ = get_feature_cols(well_embeddings, feature_type=feature_type)
    moa_pred_NSCP_per_well = nearest_neighbor_classifier_NSBW(
        well_embeddings[feature_columns],
        well_embeddings["annotation"],
        batches=well_embeddings["plate"],
        wells=well_embeddings["perturbation_id"],
        mode="NSBW",
    )
    moa_pred_NSC_per_well = nearest_neighbor_classifier_NSBW(
        well_embeddings[feature_columns],
        well_embeddings["annotation"],
        batches=well_embeddings["perturbation_id"],
        mode="NSB",
    )

    moa_pred_NSC_per_treatment = nearest_neighbor_classifier_NSBW(
        treatment_embeddings[feature_columns],
        treatment_embeddings["annotation"],
        batches=treatment_embeddings["perturbation_id"],
        mode="NSB",
    )

    pert_pert_NSPW_per_well = nearest_neighbor_classifier_NSBW(
        well_embeddings[feature_columns],
        well_embeddings["perturbation_id"],
        batches=well_embeddings["plate"],
        wells=well_embeddings["well"],
        mode="NSBW",
    )

    NSCP_accuracy_per_well = accuracy_score(
        moa_pred_NSCP_per_well, well_embeddings["annotation"]
    )
    NSC_accuracy_per_well = accuracy_score(
        moa_pred_NSC_per_well, well_embeddings["annotation"]
    )
    NSC_accuracy_per_treatment = accuracy_score(
        moa_pred_NSC_per_treatment, treatment_embeddings["annotation"]
    )
    NSPW_accuracy_per_well = accuracy_score(
        pert_pert_NSPW_per_well, well_embeddings["perturbation_id"]
    )

    return pd.DataFrame(
        {
            "model": [model_name],
            "NSC accuracy (MoA/treatment)": [NSC_accuracy_per_treatment],
            "NSC accuracy (MoA/well)": [NSC_accuracy_per_well],
            "NSCP accuracy (MoA/well)": [NSCP_accuracy_per_well],
            "NSPW accuracy (compound/well)": [NSPW_accuracy_per_well],
        }
    ).round(3)


def get_feature_importance(df, label):
    feature_columns = [c for c in df.columns if c.startswith("emb")]
    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE, class_weight="balanced"
    )
    classifier.fit(df[feature_columns], df[label])
    feature_importances = pd.DataFrame(
        {"feature": feature_columns, "importance": classifier.feature_importances_}
    ).sort_values(ascending=False, by="importance", ignore_index=True)

    feature_importances["channel"] = feature_importances["feature"].apply(
        get_feature_channel
    )
    return feature_importances


def top_feature_importance_per_treatment(df, n_features=100):
    moa_feature_importance = []
    for moa in sorted(df["annotation"].unique()):
        df[moa] = df["annotation"] == moa
        feature_importance = get_feature_importance(df, moa)
        feature_importance = feature_importance.head(n_features).value_counts("channel")
        feature_importance = pd.DataFrame(feature_importance).T
        feature_importance["annotation"] = moa
        moa_feature_importance.append(feature_importance)
    moa_feature_importance = pd.concat(moa_feature_importance)
    moa_feature_importance.columns.names = [""]
    return moa_feature_importance.reset_index(drop=True)


def get_feature_channel(entry):
    feature_number = int(entry.removeprefix("emb"))
    if feature_number < 384:
        return "DNA"
    elif 384 <= feature_number < 768:
        return "ER"
    elif 768 <= feature_number < 1152:
        return "RNA"
    elif 1152 <= feature_number < 1536:
        return "AGP"
    else:
        return "Mitochondria"


def create_plots(cell_line, embeddings):
    embeddings_cell_line = (
        embeddings[embeddings.plate.str.contains(cell_line)]
        .reset_index(drop=True)
    )
    embeddings_consensus = get_consensus_embeddings(embeddings_cell_line)
    embeddings_correlation = get_correlation_df(embeddings_consensus)
    embeddings_umap = get_umap_features(embeddings_consensus)
    sns.set_theme(font_scale=1.6)
    g = sns.clustermap(
        embeddings_correlation,
        figsize=(15, 15),
        method="average",
        metric="euclidean",
        cmap="bwr",
        vmin=-1,
        vmax=1,
        dendrogram_ratio=0.05,
        cbar_kws={"pad": 20, "orientation": "horizontal"},
        cbar_pos=(0.04, 1.01, 0.08, 0.01),
    )
    g.ax_heatmap.set_xticklabels([])
    g.ax_heatmap.set_xticks([])
    ax = g.ax_heatmap
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.savefig(IMAGES_ROOT / f"clustermap_{cell_line}.pdf")
    plt.clf()

    fontsize = 17
    sns.scatterplot(
        data=embeddings_umap,
        figsize=(8, 8),
        x="UMAP 1",
        y="UMAP 2",
        hue="annotation",
        style="annotation",
        palette="colorblind",
        s=120,
    )
    plt.legend(
        bbox_to_anchor=(0.5, -0.15),
        loc="upper center",
        borderaxespad=0,
        fontsize=fontsize,
        frameon=False,
        ncol=3,
    )
    plt.xlabel("UMAP 1", fontsize=fontsize)
    plt.ylabel("UMAP 2", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    sns.despine(top=True, right=True)
    plt.savefig(IMAGES_ROOT / f"umap_{cell_line}.pdf", bbox_inches="tight")
    plt.clf()

    embeddings_umap.to_csv(IMAGES_ROOT / f"umap_{cell_line}.csv", index=False)


def get_selected_wells(embeddings: pd.DataFrame):
    wells_plate_1_to_3 = embeddings[
        embeddings.plate.str.endswith(("P1", "P2", "P3"))
    ].query(f"well in {list(selected_concentration_plates_1_to_3['well'].values)}")
    wells_plate_4_to_6 = embeddings[
        embeddings.plate.str.endswith(("P4", "P5", "P6"))
    ].query(f"well in {list(selected_concentration_plates_4_to_6['well'].values)}")

    return pd.concat([wells_plate_1_to_3, wells_plate_4_to_6], axis=0).reset_index(
        drop=True
    )


# %%
dino_plate_1_to_3 = dino[dino.plate.str.endswith(("P1", "P2", "P3"))].query(
    f"well in {list(selected_concentration_plates_1_to_3['well'].values)}"
)
dino_plate_4_to_6 = dino[dino.plate.str.endswith(("P4", "P5", "P6"))].query(
    f"well in {list(selected_concentration_plates_4_to_6['well'].values)}"
)

dino_selected = get_selected_wells(dino)

data_batch_agg = batch_aggregate(dino)
selected_concentrations_batch_agg = batch_aggregate(dino_selected)
data_dict = {
    "all_concentrations": data_batch_agg,
    "ic50_conc": selected_concentrations_batch_agg,
}

# %%
linkages = ["average", "complete"]
for linkage in linkages:
    for conc_type in data_dict.keys():
        print(f"{linkage=}: {conc_type=}")
        batch_df = data_dict[conc_type]
        # remove trailing spaces
        batch_df["annotation"] = batch_df["annotation"].str.strip()
        feature_cols, _ = get_feature_cols(batch_df, feature_type="standard")
        batch_df["index"] = (
            + batch_df["annotation"].astype(str)
        )
        batch_df.index = batch_df["index"].values
        correlation = batch_df[feature_cols].transpose().corr()
        # plot a heatmap
        sns.set_theme(font_scale=1.7)
        clustermap = sns.clustermap(
            correlation,
            vmin=-1,
            vmax=1,
            figsize=(15, 15),
            method=linkage,
            cmap="bwr",
            dendrogram_ratio=0.05,
            cbar_kws={"pad": 20, "orientation": "horizontal"},
            cbar_pos=(0.04, 1.01, 0.08, 0.01),
        )
        clustermap.ax_heatmap.set_xticklabels([])
        clustermap.ax_heatmap.set_xticks([])

        plt.savefig(
            IMAGES_ROOT / f"{conc_type}_similaritymap_linkage={linkage}.pdf",
            bbox_inches="tight",
        )
        plt.clf()


# %%
hue_order = [
    "Antimycin A",
    "Atpenin A5",
    "Azadirachtin A",
    "Bafilomycin A1",
    "Colchicine",
    "Concanamycin A",
    "Cyclosporin A",
    "Cytochalasin D",
    "Diafenthiuron",
    "FCCP",
    "Fenbutatinoxide",
    "Latrunculin B",
    "Oligomycin A",
    "Rotenone",
    "Spirotetramat",
    "Carbofuran",
    "Paclitaxel",
    "Abamectin B1",
    "DMSO",
    "Sucrose",
]

color_df = pd.read_excel(FOLDER_ROOT / "Farbcodes.xlsx")
color_df["perturbation_id"] = pd.Categorical(
    color_df["perturbation_id"], categories=hue_order, ordered=True
)
color_df = color_df.sort_values(by="perturbation_id")
annotation = dino["annotation"].str.strip().unique()
annotation[-2], annotation[0] = annotation[0], annotation[-2]
color_df_annotation = color_df.tail(len(annotation)).copy().reset_index(drop=True)
color_df_annotation["annotation"] = annotation
color_df_annotation = (
    color_df_annotation.drop(columns="perturbation_id")
    .sort_values("annotation")
    .reset_index(drop=True)
)

dino_plot_euclidean = get_umap_features(dino, n_neighbors=20, min_dist=0.8)
dino_plot_euclidean["annotation"] = dino_plot_euclidean["annotation"].str.strip()
dino_plot_cosine = get_umap_features(
    dino, n_neighbors=15, min_dist=0.7, metric="cosine"
)
dino_plot_cosine["annotation"] = dino_plot_cosine["annotation"].str.strip()

# %%
plt.figure(figsize=(8, 8))
sns.set_style("white")
sns.scatterplot(
    data=dino_plot_euclidean,
    x="umap_x",
    y="umap_y",
    hue="perturbation_id",
    hue_order=hue_order,
    palette=list(color_df["color"].values),
    s=30,
)
sns.despine()
plt.grid(False)
plt.legend(
    bbox_to_anchor=(0.5, -0.15),
    loc="upper center",
    borderaxespad=0,
    fontsize=17,
    frameon=False,
    ncol=3,
    scatterpoints=1,
    markerscale=2,
)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.savefig(
    IMAGES_ROOT / "dino_euclidean_k=20_mindist=0.8.pdf",
    bbox_inches="tight",
)

# %%
plt.figure(figsize=(10.5, 8))
sns.set_style("white")
sns.scatterplot(
    data=dino_plot_euclidean,
    x="umap_x",
    y="umap_y",
    hue="annotation",
    hue_order=MOA_PALETTE.values(),
    palette=MOA_PALETTE.keys(),
    s=30,
)
sns.despine()
plt.grid(False)
plt.legend(
    bbox_to_anchor=(0.5, -0.15),
    loc="upper center",
    borderaxespad=0,
    fontsize=17,
    frameon=False,
    ncol=2,
    scatterpoints=1,
    markerscale=2,
)
plt.xlabel("UMAP 1", fontsize=17)
plt.ylabel("UMAP 2", fontsize=17)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.savefig(
    IMAGES_ROOT / "dino_moa_euclidean_k=20_mindist=0.8.pdf",
    bbox_inches="tight",
)

# %%
sns.set_style("white")
sns.scatterplot(
    data=dino_plot_cosine,
    x="umap_x",
    y="umap_y",
    hue="perturbation_id",
    hue_order=hue_order,
    palette=list(color_df["color"].values),
    s=10,
)
sns.despine()
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
# %%
gaussian = create_gaussian_features(dino, random_state=RANDOM_STATE)
metrics = pd.concat(
    [
        get_metrics(dino, "uniDINO"),
        get_metrics(microsnoop, "Microsnoop"),
        get_metrics(transfer, "ImageNet DINO"),
        get_metrics(cell_profiler, "CellProfiler", feature_type="cellprofiler"),
        get_metrics(random, "Random ViT"),
        get_metrics(gaussian, "Gaussian"),
    ]
).reset_index(drop=True)
metrics.to_csv(IMAGES_ROOT / "metrics.csv", index=False)

# %%
scaling_metrics = pd.concat(
    [
        get_metrics(dino_jumpcp_only, "JUMPCP only"),
        get_metrics(dino_jumpcp_hpa, "JUMPCP+HPA"),
        get_metrics(dino_jumpcp_hpa_b21, "JUMPCP+HPA+BBBC021"),
        get_metrics(dino, "uniDINO"),
    ]
).reset_index(drop=True)
#%%
scaling_metrics.to_csv(IMAGES_ROOT / "scaling_metrics.csv", index=False)

# %%
moa_feature_importance = top_feature_importance_per_treatment(
    dino_selected, n_features=50
)

sns.set_palette("colorblind")
moa_feature_importance[
    ["Mitochondria", "ER", "RNA", "AGP", "DNA", "annotation"]
].sort_values("Mitochondria").set_index("annotation").plot(kind="bar", stacked=True)
sns.despine(top=True, right=True)
plt.legend(bbox_to_anchor=(1.4, 0.5), loc="upper right", borderaxespad=0.0)
plt.xlabel("MoA")
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(IMAGES_ROOT / "feature_importance.pdf")

# %%
