from dataclasses import dataclass
import pathlib

HERE = pathlib.Path(__file__).absolute().parent
REPO_ROOT = HERE.parent
EMBEDDINGS = REPO_ROOT / "embeddings"
FIGURES = REPO_ROOT / "figures"

HCA_data_lake = pathlib.Path("/raid/cache/HCA_data_lake")


@dataclass
class BBBC021:
    root = HCA_data_lake / "BBBC021_v1"
    metadata = root / "BBBC021_v1_image.csv"
    smiles = root / "BBBC021_v1_compound.csv"
    moa = root / "BBBC021_v1_moa.csv"
    images = root / "Images"
    embeddings_root = EMBEDDINGS / "BBBC021"
    unidino_embeddings = embeddings_root / "bbbc021_unidino_mad_robustized.csv"
    transfer_embeddings = embeddings_root / "bbbc021_transfer_mad_robustized.csv"
    microsnoop_embeddings = embeddings_root / "bbbc021_microsnoop_mad_robustized.csv"
    random_vit_embeddings = embeddings_root / "bbbc021_randomvit_mad_robustized.csv"
    jcp_only_embeddings = embeddings_root / "bbbc021_jcponly_mad_robustized.csv"
    jcp_hpa_embeddings = embeddings_root / "bbbc021_jcphpa_mad_robustized.csv"
    jcp_hpa_b21_embeddings = embeddings_root / "bbbc021_jcphpab21_mad_robustized.csv"
    cell_profiler_raw = embeddings_root / "CellProfiler__well_embeddings_raw_annotated.csv"
    cell_profiler_mad_robustized = embeddings_root / "CellProfiler_well_features_mad_robustized_plate.csv"



@dataclass
class BBBC037:
    root = HCA_data_lake / "BBBC037_v1"
    ground_truth = root / "BBBC037_v1_DatasetGroundTruth.csv"
    idr = root / "gene_overexpression" / "idr0033-rohban-pathways"
    screens = idr / "screens"
    annotations = idr / "annotations"
    screen_annotations = annotations / "idr0033-screenA-annotation.csv"
    images = root / "gene_overexpression" / "images_illum_corrected"
    image_paths = root / "BBBC037_image_path.csv"
    cp = root / "CP_GeneOverexpression_dataset" / "Metadata_and_Cellprofiler_profiles"
    cp_profiles = cp / "CellPainting_overexpression_normalized_profiles__final.csv"


@dataclass
class JUMPCP:
    root = HCA_data_lake / "JUMPCP"
    sources = root / "sources"
    example_data = REPO_ROOT / "example_data"
    jumpcp_4sources_active_only = (
        sources / "dataload" / "JUMP_trainset_4sources_active_only.csv"
    )
    jumpcp_4sources_full_trainset = sources / "dataload" / "JUMP_trainset_4sources.csv"
    jumpcp_4sources_active_balanced_nontox = (
        sources
        / "dataload"
        / "JUMP_trainset_4sources_active_phenotype_balanced_nontox.csv"
    )
    jumpcp_valset_4sources = sources / "dataload" / "JUMP_valset_4sources.csv"
    embeddings_root = EMBEDDINGS / "JUMPCP"
    cell_profiler_mad_robustized = embeddings_root / "JUMPCP_cellprofiler_mad_robustized.csv"
    unidino_embeddings = embeddings_root / "JUMPCP_unidino_mad_robustized.csv"
    transfer_embeddings = embeddings_root / "JUMPCP_transfer_mad_robustized.csv"
    jcp_only_embeddings = embeddings_root / "JUMPCP_jcponly_mad_robustized.csv"
    jcp_multichannel_embeddings = embeddings_root / "JUMPCP_multichannel_mad_robustized.csv"
    microsnoop_embeddings = embeddings_root / "JUMPCP_microsnoop_mad_robustized.csv"
    random_vit_embeddings = embeddings_root / "JUMPCP_randomvit_mad_robustized.csv"


@dataclass
class HPA:
    root = HCA_data_lake / "HPA"
    kaggle = root / "kaggle_2021.tsv"
    images = root / "images"


@dataclass
class Insect:
    root = HCA_data_lake / "picasso-monheim"
    metadata = root / "metadata_all_batches_sf9.csv"
    images = root / "images"
    embeddings_root = EMBEDDINGS / "insect"
    unidino_embeddings = embeddings_root / "insect_unidino_mad_robustized.csv"
    jcp_only_embeddings = embeddings_root / "insect_jcponly_mad_robustized.csv"
    jcp_hpa_embeddings = embeddings_root / "insect_jcphpa_mad_robustized.csv"
    jcp_hpa_b21_embeddings = embeddings_root / "insect_jcphpab21_mad_robustized.csv"
    microsnoop_embeddings = embeddings_root / "insect_microsnoop_mad_robustized.csv"
    transfer_embeddings = embeddings_root / "insect_transfer_mad_robustized.csv"
    randomvit_embeddings = embeddings_root / "insect_randomvit_mad_robustized.csv"
    cellprofiler = embeddings_root / "SF9_well_profiles__normalized__feature_select.csv"


@dataclass
class CellHealth:
    root = HCA_data_lake / "Cell_Health"
    images = root / "idr0080" / "data"
    metadata = root / "metadata.csv"
    profiles = root / "profiles"
    readouts = root / "readouts"
    #  consensus_CP_features_modz = consensus / "cell_painting_modz.tsv"
    embeddings_folder = EMBEDDINGS / "cell_health" / "deep_learning"
    consensus =  EMBEDDINGS / "cell_health"  /  "cellprofiler"
    consensus_CP_features_median = consensus / "cell_painting_median.tsv"
    median_cell_health = consensus / "cell_health_median.tsv"
    results_folder = FIGURES / "cell_health" / "cell_health_metrics"