import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
import timm
from torch.utils.tensorboard import SummaryWriter

from source.io import get_concatenated_dataset

import source.dino.utils as utils
import source.dino.vision_transformer as vits
from source.dino.vision_transformer import DINOHead
from source.dino.loss import DINOLoss

import source.augment as au

torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)

def get_args_parser():
    parser = argparse.ArgumentParser("DINO", add_help=False)

    # Model parameters
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "vit_large",
            "timm",
        ]
        + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""",
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""",
    )
    parser.add_argument(
        "--out_dim",
        default=20000,
        type=int,
        help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""",
    )
    parser.add_argument(
        "--norm_last_layer",
        default=True,
        type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""",
    )
    parser.add_argument(
        "--momentum_teacher",
        default=0.996,
        type=float,
        help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""",
    )
    parser.add_argument(
        "--use_bn_in_head",
        default=False,
        type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)",
    )

    # Temperature teacher parameters
    parser.add_argument(
        "--warmup_teacher_temp",
        default=0.02,
        type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""",
    )
    parser.add_argument(
        "--teacher_temp",
        default=0.04,
        type=float,
        help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""",
    )
    parser.add_argument(
        "--warmup_teacher_temp_epochs",
        default=20,
        type=int,
        help="Number of warmup epochs for the teacher temperature (Default: 30).",
    )

    # Training/Optimization parameters
    parser.add_argument("--same_perturbations", type=utils.bool_flag, default=False)
    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=True,
        help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.04,
        help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=0.4,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=3.0,
        help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=64,
        type=int,
        help="Per-GPU batch-size : number of distinct images loaded on one GPU.",
    )
    parser.add_argument(
        "--epochs", default=200, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--freeze_last_layer",
        default=3,
        type=int,
        help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""",
    )
    parser.add_argument(
        "--lr",
        default=0.004,
        type=float,
        help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=20,
        type=int,
        help="Number of epochs for the linear learning-rate warm up.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adamw", "sgd", "lars"],
        help="""Type of optimizer. We recommend using adamw with ViTs.""",
    )
    parser.add_argument(
        "--drop_path_rate", type=float, default=0.1, help="stochastic depth rate"
    )
    parser.add_argument("--use_annot_prob", type=float, default=0.5, help="hard pos. %")
    # Multi-crop parameters
    parser.add_argument(
        "--global_crops_scale",
        type=float,
        nargs="+",
        default=(0.2, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""",
    )
    parser.add_argument(
        "--local_crops_number",
        type=int,
        default=8,
        help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """,
    )
    parser.add_argument(
        "--local_crops_scale",
        type=float,
        nargs="+",
        default=(0.05, 0.2),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""",
    )
    # Misc
    parser.add_argument(
        "--data_path",
        default="/path/to/imagenet/train/",
        type=str,
        help="Please specify path to the ImageNet training data.",
    )
    parser.add_argument(
        "--output_dir", default=".", type=str, help="Path to save logs and checkpoints."
    )
    parser.add_argument("--path_df", default="data/JUMP_trainset.csv", type=str)
    parser.add_argument(
        "--saveckp_freq", default=10, type=int, help="Save checkpoint every x epochs."
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        default=6,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore and do not set this argument.",
    )

    parser.add_argument("--include_jumpcp", default=True, type=utils.bool_flag)

    parser.add_argument("--include_bbbc021", default=True, type=utils.bool_flag)

    parser.add_argument("--include_bbbc037", default=True, type=utils.bool_flag)

    parser.add_argument("--include_hpa", default=True, type=utils.bool_flag)

    parser.add_argument("--include_insect", default=True, type=utils.bool_flag)

    parser.add_argument("--use_example_data", default=False, type=utils.bool_flag)

    parser.add_argument("--embed_dim", type=int)
    parser.add_argument("--cls_reduced_dim", default=None, type=int)
    parser.add_argument("--ckpt", type=str, default=None)
    return parser


def train_dino(args, writer=None):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    cudnn.benchmark = True
    ##############################################
    # ============ preparing data ... ============
    global_size = 224
    local_size = 96

    means = [0]
    stds = [1]
    color_prob = 1
    base_transform = au.get_default_augs(color_prob, means, stds)

    crop_global = au.RandomCropWithCells(size=global_size)
    crop_local = au.RandomCropWithCells(size=local_size)
    transforms_dino = utils.MultiScaleCrop(
        base_transform,
        crop_global,
        crop_local,
        args.local_crops_number,
        global_crops_number=2,
    )
    dataset = get_concatenated_dataset(
        transform=transforms_dino,
        include_jumpcp=args.include_jumpcp,
        include_bbbc021=args.include_bbbc021,
        include_bbbc037=args.include_bbbc037,
        include_hpa=args.include_hpa,
        include_insect=args.include_insect,
        use_example_data=args.use_example_data,
    )

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if "timm" in args.arch:
        student = timm.create_model(
            "vit_small_patch16_224", pretrained=True, in_chans=1
        )
        teacher = timm.create_model(
            "vit_small_patch16_224", pretrained=True, in_chans=1
        )
        student.head = torch.nn.Identity()
        teacher.head = torch.nn.Identity()
        embed_dim = 384
    elif args.arch in vits.__dict__.keys():
        embed_dim = args.embed_dim
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,
            embed_dim=embed_dim,
            cls_reduced_dim=args.cls_reduced_dim,
        )
        teacher = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            embed_dim=embed_dim,
            cls_reduced_dim=args.cls_reduced_dim,
        )
        if args.ckpt:
            print(f"Loading checkpoint from {args.ckpt}")
            student = utils.load_pretrained_dino(
                student,
                args.ckpt,
                "student",
                model_name=None,
                patch_size=args.patch_size,
            )
            teacher = utils.load_pretrained_dino(
                student,
                args.ckpt,
                "teacher",
                model_name=None,
                patch_size=args.patch_size,
            )
        # embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknown architecture: {args.arch}")

    # We adjust dimensionality of the head for the projection of the CLS token
    in_dim = args.cls_reduced_dim if args.cls_reduced_dim else embed_dim
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student,
        DINOHead(
            in_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(in_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[args.gpu], find_unused_parameters=False
        )
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(
        student, device_ids=[args.gpu], find_unused_parameters=False
    )
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number
        + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()


    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9
        )  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr
        * (args.batch_size_per_gpu * utils.get_world_size())
        / 256.0,  # linear scaling rule
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )
    print("Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]
    print(f"\n\n Epoch to resume {start_epoch}")

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            dino_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            args,
            writer,
        )

        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "dino_loss": dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(
                save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth")
            )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args,
    writer,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    logger = metric_logger.log_every(data_loader, 10, header)
    for it, images in enumerate(logger):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(
                images[:2]
            )  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()

        if fp16_scaler is None:
            loss.backward()
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(
                student.module.parameters(), teacher_without_ddp.parameters()
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        if utils.is_main_process():
            writer.add_scalar("Train loss step", loss, it)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if utils.is_main_process():
        writer.add_scalar(
            "Train loss epoch", metric_logger.meters["loss"].global_avg, epoch
        )
        writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar(
            "Weight decay", optimizer.param_groups[0]["weight_decay"], epoch
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINO", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if utils.is_main_process():
        writer = SummaryWriter(args.output_dir)

    with open(os.path.join(args.output_dir, "hp.json"), "wt") as f:
        json.dump(vars(args), f, indent=4)

    train_dino(args, writer)
