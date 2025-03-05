from pathlib import Path
from itertools import product
from operator import itemgetter
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from einops import rearrange

from . import image_ops as imo
from . import path as path_module
from source.utils import assert_


def repeat_rows(df: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    """
    Repeat the rows of a DataFrame a specified number of times.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose rows are to be repeated.
    multiplier : int
        The number of times each row in the DataFrame should be repeated.

    Returns
    -------
    pd.DataFrame
        A new DataFrame where each row from the input DataFrame is repeated 'multiplier' times.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> repeat_rows(df, 2)
       A  B
    0  1  3
    1  1  3
    2  2  4
    3  2  4
    """
    if multiplier < 1:
        raise ValueError("The multiplier has to be a positive integer")
    return pd.DataFrame(df.values.repeat(multiplier, axis=0), columns=df.columns)


def get_crops_with_metadata(
    image: torch.Tensor,
    meta_columns: List[str],
    data: pd.DataFrame,
    crop_size: int = 224,
    stride: int = 224,
    transform: Optional[Callable] = None,
    min_area_ratio: float = 0.01,  # Legacy code: this argument is ignored
    otsu_chan: int = 0,  # Legacy code: this argument is ignored
    otsuth: float = 0,  # Legacy code: this argument is ignored
):
    """
    Extract crops from an image and attach metadata to each crop.

    Parameters
    ----------
    image : torch.Tensor
        The image from which crops are to be generated.
    meta_columns : list of str
        The list of column names from the `data` DataFrame to be included in the metadata for each crop.
    data : pd.DataFrame
        The DataFrame containing metadata corresponding to the image.
    crop_size : int, optional
        The size of each square crop (default is 224).
    stride : int, optional
        The stride with which the crops are generated (default is 224).
    transform : callable, optional
        A function/transform that takes in a torch tensor and returns a transformed version.
    min_area_ratio : float, optional
        This argument is ignored (present for legacy reasons).
    otsu_chan : int, optional
        This argument is ignored (present for legacy reasons).
    otsuth : float, optional
        This argument is ignored (present for legacy reasons).

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'crops': A torch tensor of the image crops.
        - 'labels': A torch BoolTensor indicating the presence of a cell in each crop.
        - Additional keys corresponding to each column name in `meta_columns`, with the associated metadata.

    Notes
    -----
    The 'min_area_ratio', 'otsu_chan', and 'otsuth' parameters are present for compatibility with legacy code
    and are not used in the current implementation of the function.
    """
    crops_with_metadata = {}
    croplist, labels = imo.generate_cellcrops(
        img=image,
        crop_size=crop_size,
        min_area_ratio=min_area_ratio,
        stride=stride,
        otsu_chan=otsu_chan,
        otsuth=otsuth,
    )
    crops = torch.stack(croplist)
    if transform:
        crops = transform(crops)
    crops = rearrange(crops, "patch channel h w -> channel patch () h w")
    crops_with_metadata["crops"] = crops
    crops_with_metadata["labels"] = torch.BoolTensor(labels)
    for column in meta_columns:
        crops_with_metadata[column] = data[column]

    return crops_with_metadata


@dataclass
class DataConstants:
    """
    This class is intended to store dataset-specific constants.

    Attributes
    ----------
    jumpcp_4sources_length : int
        The length of the JUMPCP 4 sources single-channel dataset. This includes active compounds and a
        subsample of 4000 DMSO wells.

    bbbc021_length : int
        The length of the BBBC021 single-channel dataset with downsampled controls. The exact length might
        vary slightly due to random downsampling of controls.

    bbbc021_clean_length : int
        The length of the BBBC021 single-channel dataset after cleaning. The exact length might
        vary slightly due to random downsampling of controls.

    bbbc037_length : int
        The length of the BBBC037 single-channel dataset with downsampled controls. The exact length might
        vary slightly due to random downsampling of controls.

    bbbc037_clean_length : int
        The length of the BBBC037 single-channel dataset after cleaning. The exact length might
        vary slightly due to random downsampling of controls.

    hpa_length : int
        The length of the HPA single-channel training dataset

    monheim_length : int
        The length of the Monheim single-channel dataset after cleaning. The exact length might
        vary slightly due to random downsampling of controls.
    """

    jumpcp_4sources_length: int = 852_600
    bbbc021_length: int = 32_430
    bbbc021_clean_length: int = 4_720
    bbbc037_length: int = 90_140
    bbbc037_clean_length: int = 22_900
    hpa_length: int = 18_180
    monheim_length: int = 71_000



class Metadata:
    """
    This class encapsulates metadata attributes that are essential for loading and processing a dataset.
    It provides a method to validate the presence of necessary metadata attributes and columns before
    the dataset is used for training or inference.

    Attributes
    ----------
    single_channel_metadata : bool
        Indicates whether the metadata is in single-channel format.
    is_merged : bool
        Indicates whether all channels are merged in the stored image.
    metadata : pd.DataFrame
        A DataFrame containing the metadata information for the dataset.
    inference_metadata_columns : list of str
        A list of column names that are required for inference and should be present in the metadata.

    Methods
    -------
    validate()
        Validates the presence of required attributes and columns in the metadata instance. It checks for
        the existence of the `single_channel_metadata`, `is_merged`, `metadata`, and `inference_metadata_columns`
        attributes. Additionally, it ensures that the `metadata` DataFrame contains a 'path' column and all
        columns listed in `inference_metadata_columns`.

    Raises
    ------
    AttributeError
        If the Metadata instance is missing one of the required attributes.
    ValueError
        If the 'metadata' DataFrame is missing the 'path' column or any of the columns listed in
        `inference_metadata_columns`.

    Notes
    -----
    The `validate` method should be called before using the metadata instance to ensure that all required
    information is present and correctly formatted.
    """

    def validate(self):
        required_attributes = [
            "single_channel_metadata",
            "is_merged",
            "metadata",
            "inference_metadata_columns",
        ]
        for attr in required_attributes:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Metadata instance is missing required attribute '{attr}'"
                )

        metadata = getattr(self, "metadata")
        inference_metadata_columns = getattr(self, "inference_metadata_columns")

        if "path" not in metadata.columns:
            raise ValueError("The 'metadata' DataFrame must contain a 'path' column.")

        missing_columns = [
            c for c in inference_metadata_columns if c not in metadata.columns
        ]
        if missing_columns:
            raise ValueError(
                f"The 'metadata' DataFrame is missing columns: {missing_columns}"
            )


class Dataset:
    """
    A dataset class for loading and transforming image data with associated metadata.

    This class is designed to interface with a metadata object, apply transformations to the images,
    and optionally retrieve crops with metadata. It supports loading both single-channel and multi-channel
    images, with the option to handle merged or unmerged channel data.

    Parameters
    ----------
    metadata_instance : Metadata
        An instance of the Metadata class containing metadata attributes for the dataset.
    transform : Callable
        A function or callable object that applies transformations to the images.
    crops_with_metadata : bool
        A flag indicating whether to retrieve crops with metadata. Default is False.

    Attributes
    ----------
    single_channel_loading : bool
        Indicates whether the metadata should be used to load single-channel images.
    is_merged : bool
        Indicates whether the channels are stored as a single merged image.
    parsed_metadata : pd.DataFrame
        The clean metadata from the dataset.
    inference_metadata_columns : list of str
        A list of column names that are required for inference.
    transform : Callable
        The transformation function to be applied to each image.
    use_crops_with_metadata : bool
        A flag indicating whether to load metadata in addition to images.

    """

    def __init__(
        self,
        metadata_instance: Metadata,
        transform: Callable,
        use_crops_with_metadata: bool = False,
    ) -> None:
        metadata_instance.validate()
        self.single_channel_loading = metadata_instance.single_channel_metadata
        self.is_merged = metadata_instance.is_merged
        self.parsed_metadata = metadata_instance.metadata
        self.inference_metadata_columns = metadata_instance.inference_metadata_columns
        self.transform = transform
        self.crops_with_metadata = use_crops_with_metadata

    def __len__(self):
        return len(self.parsed_metadata)

    def __getitem__(self, idx: int):
        image_metadata = self.parsed_metadata.iloc[idx, :]
        image = self._load_image(
            image_metadata, self.single_channel_loading, self.is_merged
        )

        if self.crops_with_metadata:
            return get_crops_with_metadata(
                self.transform(image),
                meta_columns=self.inference_metadata_columns,
                data=image_metadata,
            )
        return self.transform(image)

    def _load_image(
        self, image_metadata: pd.DataFrame, single_channel: bool, is_merged: bool
    ):
        path = image_metadata["path"]
        if single_channel:
            # The channel number is only relevant for merged images
            channel = image_metadata.get("channel") if is_merged else None
            return self._load_single_channel_image(channel, path)
        else:
            return self._load_multi_channel_image(path, is_merged)

    def _load_single_channel_image(self, channel: Optional[int], path: str):
        image, _ = imo.load_tiff_img(path, channel=channel)
        return self._convert_to_float_tensor(image).unsqueeze(0)

    def _load_multi_channel_image(self, path: Union[str, List[str]], is_merged: bool):
        if is_merged:
            image, _ = imo.load_tiff_img(path)
            return self._convert_to_float_tensor(image)

        channels = []
        for channel_path in path:
            channel = imo.load_tiff_img(channel_path)[0]
            channel = self._convert_to_float_tensor(channel).unsqueeze(0)
            channels.append(channel)
        return torch.cat(channels)

    def _convert_to_float_tensor(self, image: np.array):
        return torch.from_numpy(imo.uint16_to_float(image))


class BBBC021Metadata(Metadata):
    num_compounds = 113
    num_moas = 13
    channels = ("DAPI", "Tubulin", "Actin")
    is_merged = False
    inference_metadata_columns = [
        "plate",
        "well",
        "batch",
        "ImageNumber",
        "compound",
        "concentration",
        "moa",
    ]
    random_state = sum([ord(char) for char in "BBBC021"])
    sampled_controls_proportion = 0.15

    def __init__(
        self,
        single_channel_metadata: bool,
        downsample_controls: bool = False,
        clean: bool = False,
        cycle: bool = False,
    ) -> None:
        self.single_channel_metadata = single_channel_metadata
        self.metadata = self._preprocessed_metadata(self.single_channel_metadata)

        if downsample_controls:
            self.metadata = self._downsample_controls(self.metadata)

        if clean:
            self.metadata = self.metadata.loc[
                self.metadata["moa"].notna(), :
            ].reset_index(drop=True)

        if cycle:
            cycle_multiplier = self._get_cycle_multiplier(clean)
            self.metadata = repeat_rows(self.metadata, cycle_multiplier)

    def _preprocessed_metadata(self, single_channel_metadata):
        smiles = pd.read_csv(path_module.BBBC021.smiles)
        moa = pd.read_csv(path_module.BBBC021.moa)
        metadata = pd.read_csv(path_module.BBBC021.metadata)
        metadata.rename(
            columns={
                "Image_Metadata_Concentration": "concentration",
                "Image_Metadata_Compound": "compound",
                "TableNumber": "batch",
                "Image_Metadata_Well_DAPI": "well",
                "Image_Metadata_Plate_DAPI": "plate",
            },
            inplace=True,
        )
        metadata = metadata.merge(smiles, how="left", on="compound")
        metadata = metadata.merge(moa, how="left", on=["compound", "concentration"])

        # In the data, there is an UNKNOWN compound with no MOA
        metadata = metadata.loc[metadata["compound"] != "UNKNOWN", :]

        if single_channel_metadata:
            # Unpivot the channels. We get a df three times as long
            value_vars = [
                "Image_FileName_DAPI",
                "Image_FileName_Actin",
                "Image_FileName_Tubulin",
            ]
            id_vars = [c for c in metadata.columns if c not in value_vars]
            metadata = (
                metadata.melt(
                    value_vars=value_vars,
                    id_vars=id_vars,
                    var_name="channel",
                    value_name="file_name",
                )
                .rename(columns={"Image_PathName_DAPI": "path_name"})
                .drop(
                    columns={"Image_PathName_Actin", "Image_PathName_Tubulin", "smiles"}
                )
            )
            path_name = metadata["path_name"].str.split("/").apply(itemgetter(-1))
            metadata["path"] = (
                str(path_module.BBBC021.images)
                + "/"
                + path_name
                + "/"
                + metadata["file_name"]
            )
            return metadata

        metadata["path"] = metadata.apply(self._generate_multichannel_paths, axis=1)
        metadata.fillna("NA", inplace=True)
        return metadata.reset_index(drop=True)

    def _downsample_controls(self, metadata):
        dmso = metadata.query("compound == 'DMSO'")  # Negative controls
        taxol = metadata.query("compound == 'taxol'")  # Positive controls
        metadata = metadata.query("compound != 'DMSO' & compound != 'taxol' ")

        dmso_downsampled, _ = train_test_split(
            dmso,
            train_size=self.sampled_controls_proportion,
            stratify=dmso["plate"],
            random_state=self.random_state,
        )

        taxol_downsampled, _ = train_test_split(
            taxol,
            train_size=self.sampled_controls_proportion,
            stratify=taxol["plate"],
            random_state=self.random_state,
        )

        return pd.concat([metadata, dmso_downsampled, taxol_downsampled]).reset_index(
            drop=True
        )

    def _generate_multichannel_paths(self, metadata):
        path_name = metadata["Image_PathName_DAPI"].split("/")[-1]
        return [
            str(path_module.BBBC021.images)
            + "/"
            + path_name
            + "/"
            + metadata[f"Image_FileName_{channel}"]
            for channel in self.channels
        ]

    def _get_cycle_multiplier(self, clean):
        if clean:
            multiplier = (
                DataConstants.jumpcp_4sources_length
                // DataConstants.bbbc021_clean_length
            )
        else:
            multiplier = (
                DataConstants.jumpcp_4sources_length // DataConstants.bbbc021_length
            )

        if multiplier == 0:
            return 1
        return multiplier


class JUMPCPMetadata(Metadata):
    is_merged = True
    inference_metadata_columns = [
        "source",
        "batch",
        "plate",
        "well",
        "perturbation_id",
        "target",
    ]

    def __init__(self, single_channel_metadata: bool, training_set: bool) -> None:
        super().__init__()
        self.single_channel_metadata = single_channel_metadata
        self.training_set = training_set
        self.metadata = self._preprocessed_metadata(
            self.single_channel_metadata, self.training_set
        )

    def _preprocessed_metadata(self, single_channel_metadata, training_set):
        if training_set:
            return self._preprocessed_training_metadata(single_channel_metadata)
        return self._preprocessed_validation_metadata(single_channel_metadata)

    def _preprocessed_training_metadata(self, single_channel_metadata):
        random_state = sum([ord(char) for char in "JUMPCP_4sources"])

        jumpcp_dmso = (
            pd.read_csv(path_module.JUMPCP.jumpcp_4sources_full_trainset)
            .query("perturbation_id == 'DMSO'")
            .groupby("source", group_keys=False)
            .apply(lambda x: x.sample(1000, random_state=random_state))
            .reset_index(drop=True)
        )
        assert_(len(jumpcp_dmso) == 4000)

        jumpcp_active = pd.read_csv(path_module.JUMPCP.jumpcp_4sources_active_only)
        metadata = pd.concat([jumpcp_dmso, jumpcp_active], axis=0).reset_index(
            drop=True
        )

        if single_channel_metadata:
            metadata = self._convert_metadata_to_single_channel(metadata)

        metadata.rename(columns={"FileName_Merged": "path"}, inplace=True)
        metadata.drop(
            columns=["Unnamed: 0", "plate_type", "index"], errors="ignore", inplace=True
        )

        # This hack is done to keep consistency on Metadata validation
        # as 'target' is part of the columns to keep during inference
        metadata["target"] = float("nan")

        return metadata

    def _preprocessed_validation_metadata(self, single_channel_metadata):
        metadata = pd.read_csv(path_module.JUMPCP.jumpcp_valset_4sources)
        if single_channel_metadata:
            raise ValueError(
                "Validation should not be done with single_channel_metadata=True!"
            )
        metadata.rename(columns={"FileName_Merged": "path"}, inplace=True)
        metadata.fillna("NA", inplace=True)
        return metadata

    def _convert_metadata_to_single_channel(self, metadata):
        metadata_length = len(metadata)
        num_channels = 5
        metadata = pd.DataFrame(
            metadata.values.repeat(num_channels, axis=0), columns=metadata.columns
        )
        metadata["channel"] = np.tile(np.arange(num_channels), metadata_length)
        return metadata

class ExampleJUMPCPMetadata(Metadata):
    is_merged = True
    inference_metadata_columns = [
        "batch",
        "plate",
        "well",
        "perturbation_id",
        "target",
    ]
    def __init__(self, single_channel_metadata: bool) -> None:
        super().__init__()
        self.single_channel_metadata = single_channel_metadata
        self.metadata = self._preprocessed_metadata(self.single_channel_metadata)

    def _preprocessed_metadata(self, single_channel_metadata):
        metadata = pd.read_csv(path_module.JUMPCP.example_data / "example_metadata.csv")
        if single_channel_metadata:
            metadata = self._convert_metadata_to_single_channel(metadata)
        metadata["path"] = path_module.REPO_ROOT / "example_data" / "JCPQC051" / metadata["path"]
        metadata.fillna("NA", inplace=True)
        return metadata

    def _convert_metadata_to_single_channel(self, metadata):
        metadata_length = len(metadata)
        num_channels = 5
        metadata = pd.DataFrame(
            metadata.values.repeat(num_channels, axis=0), columns=metadata.columns
        )
        metadata["channel"] = np.tile(np.arange(num_channels), metadata_length)
        return metadata



class BBBC037Metadata(Metadata):
    num_gene_symbols = 52
    num_gene_symbols_full = 191
    num_gene_clusters = 25
    is_merged = False
    inference_metadata_columns = [
        "well_id",
        "plate",
        "well",
        "gene_symbol",
        "has_phenotype",
        "orf_identifier",
        "moa",
    ]
    channels = ("Hoechst", "ERSyto", "ERSytoBleed", "PhGolgi", "Mito")

    random_state = sum([ord(char) for char in "BBBC037"])
    sampled_controls_proportion = 0.15

    def __init__(
        self,
        single_channel_metadata: bool,
        downsample_controls: bool = False,
        clean: bool = False,
        cycle: bool = False,
    ) -> None:
        super().__init__()

        self.metadata = self._preprocessed_metadata(single_channel_metadata)
        self.single_channel_metadata = single_channel_metadata

        if downsample_controls:
            self.metadata = self._downsample_controls(self.metadata)

        if clean:
            self.metadata = self._clean_metadata(self.metadata)

        if cycle:
            cycle_multiplier = self._get_cycle_multiplier(clean)
            self.metadata = repeat_rows(self.metadata, cycle_multiplier)

    def _preprocessed_annotations(self):
        df = pd.read_csv(path_module.BBBC037.screen_annotations)
        df.rename(
            columns={"Well Number": "WellNumber", "Plate": "PlateNumber"}, inplace=True
        )
        df = df.loc[df["PlateNumber"].str.contains("_illum_corrected"), :]
        df["PlateNumber"] = (
            df["PlateNumber"].str.replace("_illum_corrected", "").astype("int64")
        )

        df["Gene Symbol"].fillna("control", inplace=True)

        return df

    def _preprocessed_metadata(self, single_channel_metadata):
        ground_truth = pd.read_csv(path_module.BBBC037.ground_truth).reset_index()
        annotations = self._preprocessed_annotations()
        path_df = pd.read_csv(path_module.BBBC037.image_paths)
        metadata = path_df.merge(
            annotations, how="left", on=["PlateNumber", "WellNumber"]
        )
        metadata = metadata.merge(
            ground_truth,
            how="left",
            left_on="ORF Identifier",
            right_on="Metadata_broad_sample",
        )
        if single_channel_metadata:
            value_vars = [
                "ImagePath_Hoechst",
                "ImagePath_ERSyto",
                "ImagePath_ERSytoBleed",
                "ImagePath_PhGolgi",
                "ImagePath_Mito",
            ]
            id_vars = [c for c in metadata.columns if c not in value_vars]
            metadata = metadata.melt(
                value_vars=value_vars,
                id_vars=id_vars,
                var_name="channel",
                value_name="path",
            )
            metadata["path"] = (
                str(path_module.BBBC037.images)
                + "/"
                + metadata["PlateNumber"].astype(str)
                + "/"
                + metadata["path"]
            )
        else:
            metadata["path"] = metadata.apply(self._create_multi_channel_paths, axis=1)

        metadata.rename(
            columns={
                "WellNumber": "well_id",
                "PlateNumber": "plate",
                "Well": "well",
                "Gene Symbol": "gene_symbol",
                "Has Phenotype": "has_phenotype",
                "ORF Identifier": "orf_identifier",
                "Metadata_moa": "moa",
            },
            inplace=True,
        )

        metadata = metadata[
            [
                "well_id",
                "plate",
                "well",
                "gene_symbol",
                "has_phenotype",
                "orf_identifier",
                "moa",
                "path",
            ]
        ]

        metadata.fillna("NA", inplace=True)

        return metadata

    def _downsample_controls(self, metadata):
        controls = metadata.loc[metadata["gene_symbol"] == "control", :]
        controls_downsampled, _ = train_test_split(
            controls,
            train_size=self.sampled_controls_proportion,
            stratify=controls["plate"],
            random_state=self.random_state,
        )
        metadata = metadata.loc[metadata["gene_symbol"] != "control", :]

        return pd.concat([metadata, controls_downsampled]).reset_index(drop=True)

    def _clean_metadata(self, metadata):
        has_phenotype = metadata["has_phenotype"] == "yes"
        is_control = metadata["gene_symbol"] == "control"
        metadata = metadata.loc[has_phenotype | is_control, :]
        metadata = metadata.reset_index(drop=True)
        return metadata

    def _get_cycle_multiplier(self, clean):
        if clean:
            multiplier = (
                DataConstants.jumpcp_4sources_length
                // DataConstants.bbbc037_clean_length
            )
        else:
            multiplier = (
                DataConstants.jumpcp_4sources_length // DataConstants.bbbc037_length
            )

        if multiplier == 0:
            return 1
        return multiplier

    def _create_multi_channel_paths(self, image):
        paths = []
        for channel in self.channels:
            prefix = f"{str(path_module.BBBC037.images)}/{str(image['PlateNumber'])}/"
            image_path = image[f"ImagePath_{channel}"]
            paths.append(prefix + image_path)
        return paths


class CellHealthMetadata(Metadata):
    channels = {1: "DNA", 2: "ER", 3: "RNA", 4: "AGP", 5: "Mito"}
    is_merged = False
    inference_metadata_columns = [
        "plate",
        "well",
        "cell_line",
        "reagent_identifier",
        "gene_symbol",
        "control_type",
        "comment",
    ]

    def __init__(self, single_channel_metadata: bool) -> None:
        super().__init__()
        self.metadata = self._preprocessed_metadata()
        if single_channel_metadata:
            raise NotImplementedError(
                "Metadata preprocessing is only implemented for multi-channel data loading!"
            )
        self.single_channel_metadata = single_channel_metadata

    def _preprocessed_metadata(self):
        metadata = pd.read_csv(path_module.CellHealth.metadata)
        metadata = metadata.astype({"row": str, "column": str, "field": str})
        metadata["row"] = metadata["row"].str.zfill(2)
        metadata["column"] = metadata["column"].str.zfill(2)
        metadata["field"] = metadata["field"].str.zfill(2)
        metadata["path"] = metadata.apply(self._create_multi_channel_paths, axis=1)
        metadata.fillna("NA", inplace=True)
        return metadata

    def _create_multi_channel_paths(self, image):
        paths = []
        for channel in self.channels.keys():
            prefix = f"{str(path_module.CellHealth.images)}/{image['plate']}/Images/"
            image_path = f"r{image['row']}c{image['column']}f{image['field']}p01-"
            channel_suffix = f"ch{channel}sk1fk1fl1.tiff"
            paths.append(prefix + image_path + channel_suffix)
        return paths


class HPAMetadata(Metadata):
    organelle_lookup = {
        "Nucleoplasm": 0,
        "Nuclear membrane": 1,
        "Nucleoli": 2,
        "Nucleoli fibrillar center": 3,
        "Nuclear speckles": 4,
        "Nuclear bodies": 5,
        "Endoplasmic reticulum": 6,
        "Golgi apparatus": 7,
        "Intermediate filaments": 8,
        "Actin filaments": 9,
        "Focal adhesion sites": 9,
        "Microtubules": 10,
        "Mitotic spindle": 11,
        "Centrosome": 12,
        "Centriolar satellite": 12,
        "Plasma membrane": 13,
        "Cell Junctions": 13,
        "Mitochondria": 14,
        "Aggresome": 15,
        "Cytosol": 16,
        "Vesicles": 17,
        "Peroxisomes": 17,
        "Endosomes": 17,
        "Lysosomes": 17,
        "Lipid droplets": 17,
        "Cytoplasmic bodies": 17,
        "No staining": 18,
    }
    channels = ["blue", "red", "green", "yellow"]

    num_cell_lines = 17
    num_organelles = 27

    is_merged = False
    inference_metadata_columns = [
        "image",
        "cell_line",
        "label",
    ]

    def __init__(self, single_channel_metadata: bool, cycle: bool = False) -> None:
        super().__init__()
        self.metadata = self._preprocessed_metadata(single_channel_metadata)
        self.single_channel_metadata = single_channel_metadata
        if cycle:
            cycle_multiplier = self._get_cycle_multiplier()
            self.metadata = repeat_rows(self.metadata, cycle_multiplier)

    def _preprocessed_metadata(self, single_channel_metadata: bool):
        metadata = pd.read_csv(path_module.HPA.kaggle)

        metadata = metadata[metadata["in_trainset"]].dropna().reset_index(drop=True)
        metadata["label_idx"] = self._label_to_list(metadata["Label_idx"])
        metadata.drop(columns=["Label_idx", "in_trainset"], inplace=True)
        metadata.rename(
            columns={"Image": "image", "Label": "label", "Cellline": "cell_line"},
            inplace=True,
        )
        metadata["image"] = metadata["image"].apply(lambda p: Path(p).name)
        metadata_length = len(metadata)

        if single_channel_metadata:
            metadata = pd.DataFrame(
                metadata.values.repeat(len(self.channels), axis=0),
                columns=metadata.columns,
            )
            metadata["channel"] = np.tile(np.array(self.channels), metadata_length)
            metadata["path"] = (
                str(path_module.HPA.images)
                + "/"
                + metadata["image"]
                + "_"
                + metadata["channel"]
                + ".tif"
            )

            return metadata

        metadata["path"] = metadata.apply(self._create_multi_channel_paths, axis=1)

        return metadata

    def _label_to_list(self, labels):
        labels = labels.str.split("|")
        labels = labels.apply(lambda ls: [int(e) for e in ls])
        return labels

    def _label_one_hot(self, label_idx):
        array = np.zeros(len(set(self.organelle_lookup)))
        for idx in label_idx:
            array[idx] = 1.0
        return array.astype(np.float32)

    def _create_multi_channel_paths(self, image):
        paths = []
        for channel in self.channels:
            prefix = str(path_module.HPA.images)
            image_path = f"/{image['image']}_{channel}.tif"
            paths.append(prefix + image_path)
        return paths

    def _get_cycle_multiplier(self):
        multiplier = DataConstants.jumpcp_4sources_length // DataConstants.hpa_length

        if multiplier == 0:
            return 1

        return multiplier


# %%
class InsectMetadata(Metadata):
    channels = {1: "DNA", 2: "ER", 3: "RNA", 4: "AGP", 5: "Mito"}
    num_perturbations = 19
    is_merged = False
    inference_metadata_columns = [
        "plate",
        "well",
        "cell_count",
        "batch",
        "perturbation_id",
        "MoA",
        "concentration",
    ]

    def __init__(
        self,
        single_channel_metadata: bool,
        downsample_controls: bool = False,
        cycle: bool = False,
    ) -> None:
        super().__init__()
        self.metadata = self._preprocessed_metadata(single_channel_metadata)
        self.single_channel_metadata = single_channel_metadata

        if downsample_controls:
            self.metadata = self._downsample_controls(self.metadata)

        if cycle:
            cycle_multiplier = self._get_cycle_multiplier()
            self.metadata = repeat_rows(self.metadata, cycle_multiplier)

    def _preprocessed_metadata(self, single_channel_metadata):
        metadata = pd.read_csv(path_module.Insect.metadata).query("batch == 'Batch3'")
        if single_channel_metadata:
            metadata_length = len(metadata)
            metadata = pd.DataFrame(
                metadata.values.repeat(len(self.channels), axis=0),
                columns=metadata.columns,
            )
            metadata["channel"] = np.tile(
                np.array(tuple(self.channels.keys())), metadata_length
            )
            metadata["path"] = (
                str(path_module.Insect.images)
                + "/"
                + metadata["batch"]
                + "/"
                + metadata["plate"]
                + "/Images/"
                + metadata["fov_id"]
                + "p01-ch"
                + metadata["channel"].astype(str)
                + "sk1fk1fl1.tiff"
            )
            return metadata.reset_index(drop=True)

        metadata["path"] = metadata.apply(self._create_multi_channel_paths, axis=1)

        return metadata.reset_index(drop=True)

    def _downsample_controls(self, metadata):
        dmso = metadata.query("perturbation_id == 'DMSO'")
        metadata = metadata.query("perturbation_id != 'DMSO'")

        dmso_downsampled, _ = train_test_split(
            dmso,
            train_size=0.10,
            stratify=dmso.plate,
            random_state=100,
        )

        return pd.concat([metadata, dmso_downsampled]).reset_index(drop=True)

    def _create_multi_channel_paths(self, image):
        paths = []
        for channel in self.channels.keys():
            prefix = (
                str(path_module.Insect.images)
                + "/"
                + image["batch"]
                + "/"
                + image["plate"]
                + "/Images/"
            )
            image_path = image["fov_id"] + "p01-ch" + str(channel) + "sk1fk1fl1.tiff"
            paths.append(prefix + image_path)
        return paths

    def _get_cycle_multiplier(self):
        multiplier = (
            DataConstants.jumpcp_4sources_length // DataConstants.monheim_length
        )

        if multiplier == 0:
            return 1

        return multiplier


# %%
def get_concatenated_dataset(
    transform,
    include_jumpcp=True,
    include_bbbc021=True,
    include_bbbc037=True,
    include_hpa=True,
    include_insect=True,
    use_example_data=False,
):
    if use_example_data:
        metadata = ExampleJUMPCPMetadata(single_channel_metadata=True)
        return Dataset(metadata_instance=metadata, transform=transform)

    cycle = True if include_jumpcp else False

    jumpcp = JUMPCPMetadata(single_channel_metadata=True, training_set=True)
    bbbc021 = BBBC021Metadata(
        single_channel_metadata=True, downsample_controls=True, clean=False, cycle=cycle
    )
    bbbc037 = BBBC037Metadata(
        single_channel_metadata=True, downsample_controls=True, clean=False, cycle=cycle
    )
    hpa = HPAMetadata(single_channel_metadata=True, cycle=cycle)
    insect = InsectMetadata(
        single_channel_metadata=True, downsample_controls=True, cycle=cycle
    )

    datasets = []
    if include_jumpcp:
        datasets.append(jumpcp)
    if include_bbbc021:
        datasets.append(bbbc021)
    if include_bbbc037:
        datasets.append(bbbc037)
    if include_hpa:
        datasets.append(hpa)
    if include_insect:
        datasets.append(insect)


    datasets = [
        Dataset(metadata_instance=dataset, transform=transform) for dataset in datasets
    ]

    return torch.utils.data.ConcatDataset(datasets)
