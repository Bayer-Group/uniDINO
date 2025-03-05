# %%
from pathlib import Path
import argparse
import pycytominer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

import pandas as pd
from sklearn.preprocessing import normalize

from source.inference_utils import (
    forward_inference,
    aggregate_embeddings_plate,
)
import source.io as io

from source.dino.utils import load_pretrained_dino
import source.dino.vision_transformer as vits
from source.microsnoop import MicrosnoopCustomWrapper
from source.utils import get_feature_cols, assert_


# %%
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Self-supervised model",
        default="dino",
        choices=["dino", "random_vit", "transfer", "microsnoop"],
        type=str,
    )
    parser.add_argument(
        "--arch",
        default="vit_small_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train, used only for MAE",
    )
    parser.add_argument("--ckpt", help="Model checkpoint file", type=str)
    parser.add_argument(
        "--valset",
        help="Validation set",
        default="jumpcp-valset",
        choices=[
            "jumpcp-valset",
            "cell-health",
            "BBBC021",
            "HPA",
            "BBBC037",
            "insect",
            "example_data"
        ],
        type=str,
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--cls_reduced_dim", default=None, type=int)
    parser.add_argument(
        "--num_workers", type=int, default=6, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--operation",
        help="How to aggregate crops, FOV, and perturbations",
        default="mean",
        choices=["mean", "median", "none"],
        type=str,
    )
    parser.add_argument(
        "--norm_method",
        help="How to norm features",
        default="all",
        choices=[
            "standardize",
            "mad_robustize",
            "spherize",
            "no_post_proc",
        ],
        type=str,
    )
    parser.add_argument("-o", "--outdir", help="Output directory")
    parser.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--size", nargs="?", const=224, type=int, default=224)
    parser.add_argument("--stride", nargs="?", const=None, type=int, default=None)
    parser.add_argument("--l2norm", default=False, action="store_true")
    parser.add_argument("--embed_dim", type=int)
    return parser


def get_model(args, device):
    if args.model == "dino":
        embed_dim = args.embed_dim
        patch_size = 16
        checkpoint_key = "teacher"
        arch = "vit_small" if "small" in args.arch else "vit_base"
        arch = (
            arch.split("_")[0] + "_x" + arch.split("_")[1] if "x" in args.arch else arch
        )
        model = vits.__dict__[arch](
            patch_size=patch_size,
            drop_path_rate=0.1,
            in_channels=1,
            embed_dim=embed_dim,
            cls_reduced_dim=args.cls_reduced_dim,
        )
        embednet = load_pretrained_dino(
            model, args.ckpt, checkpoint_key, model_name=None, patch_size=patch_size
        )
    elif args.model == "random_vit":
        embed_dim = args.embed_dim
        patch_size = 16
        arch = "vit_small" if "small" in args.arch else "vit_base"
        embednet = vits.__dict__[arch](
            patch_size=patch_size,
            drop_path_rate=0.1,
            in_channels=1,
            embed_dim=embed_dim,
            cls_reduced_dim=args.cls_reduced_dim,
        )
    elif args.model == "transfer":

        class Adapter(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def forward(self, x):
                return x.repeat_interleave(3, dim=1)

        embednet = timm.create_model(
            "hf_hub:timm/vit_small_patch16_224.dino", pretrained=True, num_classes=0
        )
        embednet = torch.nn.Sequential(Adapter(), embednet)
    elif args.model == "microsnoop":
        embednet = MicrosnoopCustomWrapper()
    else:
        raise ValueError(
            "Invalid --model argument. Should be one of: [simclr, dino, mae]"
        )

    model = (
        torch.nn.DataParallel(embednet, device_ids=args.gpus)
        if torch.cuda.is_available()
        else embednet
    )
    model.to(device)
    model.eval()
    return model


def get_dataset(args):
    transform = []

    if args.valset == "jumpcp-valset":
        metadata = io.JUMPCPMetadata(single_channel_metadata=False, training_set=False)
    if args.valset == "cell-health":
        # Without making the images smaller, inference takes too long
        transform.append(transforms.Resize(1080))
        metadata = io.CellHealthMetadata(single_channel_metadata=False)
    if args.valset == "BBBC021":
        metadata = io.BBBC021Metadata(single_channel_metadata=False)
    if args.valset == "BBBC037":
        metadata = io.BBBC037Metadata(single_channel_metadata=False)
    if args.valset == "HPA":
        # Without making the images smaller, inference takes too long
        # There are two different resolutions in the data set
        transform.append(transforms.Resize(1080))
        metadata = io.HPAMetadata(single_channel_metadata=False)
    if args.valset == "insect":
        metadata = io.InsectMetadata(single_channel_metadata=False)
    if args.valset == "example_data":
        metadata = io.ExampleJUMPCPMetadata(single_channel_metadata=False)

    return io.Dataset(
        metadata_instance=metadata,
        transform=transforms.Compose(transform),
        use_crops_with_metadata=True,
    )


args = get_parser().parse_args()
print(f"Inference for {args.valset} dataset.")
print(f"Norm method {args.norm_method}")
norm_method = [args.norm_method]
crop_size = args.size
stride = args.stride if args.stride is not None else crop_size
assert_(Path(args.outdir).exists())
outdirs = [Path(args.outdir)]

dataset = get_dataset(args)
device = torch.device(
    "cuda:" + str(args.gpus[0]) if torch.cuda.is_available() else "cpu"
)
model = get_model(args, device)
metadata = dataset.parsed_metadata


dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    drop_last=False,
)
embeddings = []
for crops_with_metadata in tqdm(dataloader):
    with torch.no_grad():
        crop_embeddings = forward_inference(
            model,
            crops_with_metadata["crops"],
            crops_with_metadata["labels"],
            device,
        )
        embeddings.append(crop_embeddings.cpu().numpy())

embedding_df = aggregate_embeddings_plate(
    plate_dfr=dataset.parsed_metadata,
    plate_embs=embeddings,
    my_cols=dataset.inference_metadata_columns,
    operation=args.operation,
)

print(
    f"Postprocessing embeddings method: {norm_method},  aggregation: {args.operation}"
)
assert_(isinstance(norm_method, list), f"norm_method {norm_method} is not a list")

def postprocess_embeddings(
    profiles,
    model="CellProfiler",
    norm_method="spherize",
    l2_norm=False,
    grouping_feature="plate",
):
    if norm_method == "no_post_proc":
        return profiles

    profiles[grouping_feature] = profiles[grouping_feature].astype(str)
    feature_type = "cellprofiler" if model == "CellProfiler" else "standard"
    embedding_features, meta_features = get_feature_cols(
        profiles, feature_type=feature_type
    )

    if l2_norm:
        profiles[embedding_features] = normalize(
            profiles[embedding_features], axis=0, norm="l2"
        )
    if norm_method == "no_post_proc":
        return profiles

    embedding_df = profiles.loc[:, embedding_features]
    meta_df = profiles.loc[:, meta_features]

    profiles = pd.concat([meta_df, embedding_df], axis=1).reset_index()

    if norm_method == "spherize":
        normalized_profiles = pycytominer.normalize(
            profiles=profiles,
            features=embedding_features,
            meta_features=meta_features,
            samples="all",
            method=norm_method,
            output_file=None,
        )
    elif norm_method in ["standardize", "mad_robustize", "robustize"]:
        normalized_profiles = []
        for group in profiles[grouping_feature].unique():
            group_df = (
                profiles.query(f"{grouping_feature} == '{group}'")
                .reset_index(drop=True)
                .copy()
            )

            group_df = pycytominer.normalize(
                profiles=group_df,
                features=embedding_features,
                meta_features=meta_features,
                samples="all",
                method=norm_method,
                output_file=None,
            )

            normalized_profiles.append(group_df)

        normalized_profiles = pd.concat(normalized_profiles).reset_index(drop=True)
    else:
        raise ValueError(f"Invalid {norm_method=} specified")

    return normalized_profiles


postprocessed_embeddings = postprocess_embeddings(
    profiles=embedding_df,
    model="standard",
    norm_method=args.norm_method,
    l2_norm=args.l2norm
)

csv_path_well = f"{args.outdir}/{args.valset}_{args.norm_method}_well_features.csv"
postprocessed_embeddings.to_csv(csv_path_well, index=False)
print("Wrote well and consensus embeddings")
