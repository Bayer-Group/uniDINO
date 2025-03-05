# %%
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from pathlib import Path
from joblib import Parallel, delayed

from source.eval import create_gaussian_features
from source import path

RANDOM_STATE = np.random.RandomState(5432)

RESULTS_FOLDER = path.CellHealth.results_folder
EMBEDDINGS_FOLDER = path.CellHealth.embeddings_folder

consensus_features_cell_profiler = pd.read_csv(
    path.CellHealth.consensus_CP_features_median, sep="\t"
)
normalized_cell_health = pd.read_csv(path.CellHealth.median_cell_health, sep="\t")

top_performing_tasks = [
    "vb_live_cell_width_length",
    "vb_num_live_cells",
    "vb_percent_dead_only",
    "vb_live_cell_roundness",
    "cc_s_n_objects",
    "cc_cc_n_objects",
    "vb_percent_live",
    "vb_percent_dead",
    "cc_s_intensity_nucleus_area_sum",
    "cc_g1_n_objects",
    "cc_all_n_objects",
    "cc_cc_s",
    "vb_live_cell_area",
    "cc_g1_plus_g2_count",
    "cc_g1_n_spots_h2ax_mean",
]

metadata_columns = ["Metadata_profile_id", "Metadata_pert_name", "Metadata_cell_line"]


# %%
def get_consensus_profiles(embeddings, single_fov=False):
    if single_fov:
        embeddings = embeddings.query(f"field == {single_fov}").drop(
            columns={
                "id",
                "plate_id",
                "cell_line_index",
                "field",
                "column",
                "row",
                "well_id",
                "gene_symbol_index",
            }
        )
    embeddings = (
        embeddings.groupby(["gene_symbol", "cell_line", "reagent_identifier"])
        .mean(numeric_only=True)
        .reset_index()
    )
    return embeddings


def load_embedding(embedding_path: Path):
    print(f"Loading {embedding_path}")
    if embedding_path.suffix == ".parquet":
        embedding = pd.read_parquet(embedding_path)
    elif embedding_path.suffix == ".csv":
        embedding = pd.read_csv(embedding_path)
    else:
        raise NotImplementedError(f"{embedding_path.suffix} not supported!")

    embedding["reagent_identifier"] = [
        gene if reagent is None else reagent
        for reagent, gene in zip(
            embedding["reagent_identifier"],
            embedding["gene_symbol"],
        )
    ]
    embedding.drop(columns="Unnamed: 0", inplace=True, errors="ignore")
    return embedding


def shuffle_features(features):
    features_cols = [c for c in features.columns if not c.startswith("Meta")]
    shuffled_features = features.copy()
    for col in features_cols:
        shuffled_features[col] = (
            shuffled_features[col].sample(len(shuffled_features)).reset_index(drop=True)
        )
    return shuffled_features


def get_scores(consensus_features, cell_health, predictive_score, n_splits=5):
    def get_scores_kfold(X, y, predictive_score, n_splits):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            score = predictive_score(X_train, X_test, y_train, y_test)
            scores.append(score)

        return np.mean(scores)

    def process_readout(cell_line, readout, n_splits):
        y = cell_health.query(f"Metadata_cell_line == '{cell_line}'")[readout]
        selected_rows = y.notna()
        X_clean, y_clean = X.loc[selected_rows, :], y[selected_rows]
        return readout, get_scores_kfold(X_clean, y_clean, predictive_score, n_splits)

    feature_cols_profiles = [
        c for c in consensus_features.columns if not c.startswith("Metadata")
    ]
    feature_cols_readouts = [
        c for c in cell_health.columns if not c.startswith("Metadata")
    ]
    cell_lines = consensus_features["Metadata_cell_line"].unique()
    scores = {cell_line: {} for cell_line in cell_lines}
    for cell_line in cell_lines:
        print(f"Cell line: {cell_line}")
        X = consensus_features.query(f"Metadata_cell_line == '{cell_line}'")
        X = X[feature_cols_profiles]

        readout_scores = Parallel(n_jobs=len(feature_cols_readouts) // 2)(
            delayed(process_readout)(cell_line, readout, n_splits)
            for readout in feature_cols_readouts
        )

        scores[cell_line] = {readout: score for readout, score in readout_scores}

    scores = pd.DataFrame(scores)
    scores["readout"] = scores.index
    scores = scores.reset_index(drop=True)
    return scores


def knn_predictive_score(X_train, X_test, y_train, y_test, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    fit = model.fit(X_train, y_train)
    return fit.score(X_test, y_test)


def random_forest_predictive_score(
    X_train, X_test, y_train, y_test, min_samples_leaf=0.01
):
    model = RandomForestRegressor(min_samples_leaf=min_samples_leaf, n_estimators=100)
    fit = model.fit(X_train, y_train)
    return fit.score(X_test, y_test)


def rsquared_cumulative(scores, step_size=0.005):
    thresholds = np.arange(0, 1, step_size)
    return np.array([sum(scores.values > th) / len(scores) for th in thresholds])


def align_features_and_readouts(consensus_profiles):
    merged_consensus = (
        consensus_profiles.rename(
            columns={
                "reagent_identifier": "Metadata_pert_name",
                "cell_line": "Metadata_cell_line",
            }
        )
        .merge(normalized_cell_health)
        .drop(columns=["gene_symbol"])
    )
    features_consensus = merged_consensus.filter(regex=r"^(emb|Meta)")
    readouts = merged_consensus[
        [c for c in merged_consensus.columns if not c.startswith("emb")]
    ]

    return features_consensus, readouts


def plot_rsquared_curve(scores, figsize=(5, 10)):
    sns.set_palette(sns.color_palette("colorblind"))
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=figsize)
    fig.suptitle("Cell Health Predictive Score")

    for _, score in scores.items():
        step_size = 1 / len(score)
        x = np.arange(0, 1, step_size)
        cell_lines = score.columns
        for i, cell_line in enumerate(cell_lines):
            axs[i].plot(x, score.values[:, i])
            axs[i].set_title(cell_line)
    plt.legend(scores.keys())
    plt.xlabel("R-squared")


def validate_embeddings(embeddings_folder):
    embeddings = {
        p.stem.replace("_", " "): load_embedding(p)
        for p in embeddings_folder.iterdir()
        if p.is_file()
    }
    scores = scores_pipeline(embeddings)
    cumulative_scores = {}
    score_dfs = []
    for key, score in scores.items():
        scores_no_readout = score.drop(columns="readout")
        cumulative_scores[key] = pd.DataFrame(
            rsquared_cumulative(scores_no_readout), columns=scores_no_readout.columns
        )
        score["model"] = key
        score_dfs.append(score)

    return cumulative_scores, pd.concat(score_dfs, axis=0)


def scores_pipeline(embeddings, model="random_forest"):
    predictive_score = (
        random_forest_predictive_score
        if model == "random_forest"
        else knn_predictive_score
    )
    consensus_embeddings = {
        key: get_consensus_profiles(emb) for key, emb in embeddings.items()
    }
    # Add baselines

    aligned_embeddings = {
        key: align_features_and_readouts(emb)
        for key, emb in consensus_embeddings.items()
    }

    aligned_embeddings["Cell Profiler"] = (
        consensus_features_cell_profiler,
        normalized_cell_health,
    )

    aligned_embeddings["Gaussian"] = (
        create_gaussian_features(
            consensus_features_cell_profiler, random_state=RANDOM_STATE
        ),
        normalized_cell_health,
    )

    n_jobs = len(aligned_embeddings)

    scores = Parallel(n_jobs=n_jobs)(
        delayed(get_scores)(features, readout, predictive_score, n_splits=5)
        for key, (features, readout) in aligned_embeddings.items()
    )
    return {key: score for key, score in zip(aligned_embeddings.keys(), scores)}


def calculate_auc(scores):
    x = np.arange(0, 1, 1 / len(scores))
    return np.trapz(scores, x)


# %%

if not RESULTS_FOLDER.exists():
    RESULTS_FOLDER.mkdir()

# %%
start_time = time.time()
cumulative_scores, scores = validate_embeddings(EMBEDDINGS_FOLDER)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time of validation pipeline: {elapsed_time:.2f} seconds")
# %%
plot_rsquared_curve(cumulative_scores)
plt.savefig(RESULTS_FOLDER / "cumulative_rsquared.png")
plt.clf()

scores.to_csv(RESULTS_FOLDER / "r_square_raw.csv", index=False)

# %%
auc_scores = []
for model, cumulative_score in cumulative_scores.items():
    auc_score = pd.DataFrame(
        {"AUC": cumulative_score.apply(calculate_auc)}
    )
    auc_score["cell_line"] = auc_score.index
    auc_score = auc_score.reset_index(drop=True)
    auc_score["model"] = model
    auc_scores.append(auc_score)

pd.concat(auc_scores, axis=0).round(3).to_csv(
    RESULTS_FOLDER / "r_squared_AUC.csv", index=False
)
# %%
