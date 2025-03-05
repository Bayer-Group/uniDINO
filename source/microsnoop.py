# Code adapted from https://github.com/cellimnet/microsnoop-publish
#%%
import math
import os
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch
import h5py
import torch.distributed as dist
from scellseg.net_utils import downsample as CNN_downsample
from scellseg.net_utils import upsample as CNN_upsample
from scellseg.transforms import make_tiles

from source.utils import assert_


class CoreModel:
    def __init__(self):
        rank = get_rank()
        if rank == 0:
            print(">>>> Model Init Start")

    def _embed_imgs(self, X, y, chan, args, embed_fn, tile=False):
        # set data sample and loader
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X * 1.0).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if isinstance(chan, np.ndarray):
            chan = torch.from_numpy(chan)
        data_sampler = torch.utils.data.DistributedSampler(
            X, num_replicas=get_world_size(), rank=get_rank(), shuffle=False
        )
        code = torch.range(
            0, len(X) - 1, dtype=torch.int32
        )  # Note: 是为了查找哪些数据重复了, 方便后面删除
        dataset_pair = torch.utils.data.TensorDataset(X, y, chan, code)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset_pair,
            drop_last=False,
            sampler=data_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        global_rank = get_rank()

        # for each batch
        embeddings, ys, chans, codes = None, None, None, None
        self.net.eval()
        for ibatch, (Xi, yi, chani, codei) in enumerate(data_loader):
            # print('ibatch， len(Xi), rank', ibatch, len(Xi), get_rank())

            # run net
            Xi = Xi.cuda(non_blocking=True)
            latent = self.net.module.encoder(Xi, mask_ratio=0)[
                0
            ]  # transformor网络拿的是最后一层中的cls_token
            embedding = embed_fn(latent)

            gathered_embedding = [
                torch.empty_like(embedding) for _ in range(args.world_size)
            ]
            yi = yi.cuda(non_blocking=True)
            chani = chani.cuda(non_blocking=True)
            codei = codei.cuda(non_blocking=True)
            gathered_yi = [torch.empty_like(yi) for _ in range(args.world_size)]
            gathered_chani = [torch.empty_like(chani) for _ in range(args.world_size)]
            gathered_codei = [torch.empty_like(codei) for _ in range(args.world_size)]
            dist.barrier()  # tile模式下 先把多个GPU上的结果汇总起来，不然同一张大图上的数据就会被分成四份
            dist.all_gather(gathered_embedding, embedding.contiguous())  # contiguous()
            dist.all_gather(gathered_yi, yi.contiguous())  # contiguous()
            dist.all_gather(gathered_chani, chani.contiguous())  # contiguous()
            dist.all_gather(gathered_codei, codei.contiguous())  # contiguous()
            embedding = [
                embeddingi.cpu().detach().numpy()
                for gathered_embeddingi in gathered_embedding
                for embeddingi in gathered_embeddingi
            ]
            yi = [
                yij.cpu().detach().numpy()
                for gathered_yij in gathered_yi
                for yij in gathered_yij
            ]
            chani = [
                chanij.cpu().detach().numpy()
                for gathered_chanij in gathered_chani
                for chanij in gathered_chanij
            ]
            codei = [
                codeij.cpu().detach().numpy()
                for gathered_codeij in gathered_codei
                for codeij in gathered_codeij
            ]

            if global_rank == args.local_rank:
                if embeddings is None:
                    embeddings = np.array(embedding)
                else:
                    embeddings = np.concatenate(
                        (embeddings, np.array(embedding)), axis=0
                    )
                if ys is None:
                    ys = yi
                    chans = chani
                    codes = codei
                else:
                    ys = np.concatenate((ys, yi), axis=0)
                    chans = np.concatenate((chans, chani), axis=0)
                    codes = np.concatenate((codes, codei), axis=0)
        torch.cuda.synchronize()  # 时间相关
        if global_rank == args.local_rank:
            codes = [str(codeii) for codeii in codes]
            code_df = pd.DataFrame(
                {
                    "embeddings": list(embeddings),
                    "ys": list(ys),
                    "chans": list(chans),
                    "codes": codes,
                }
            )
            code_df_drop = (
                code_df.groupby(by=["codes"])
                .sample(n=1)
                .sort_values("codes", ascending=True)
            )
            embeddings = np.array(code_df_drop.embeddings.to_list())
            ys = code_df_drop.ys.to_numpy()
            chans = code_df_drop.chans.to_numpy()
            if tile:
                embeddings = embeddings.mean(
                    axis=0
                )  # Note 同一张图上的embedding聚合方式，这里就简单的求了下平均
                ys = ys[0]  # tile模式下默认一样的
                chans = chans[0]  # tile模式下默认一样的

            # print('len(embeddings)&len(ys)', len(ys), len(embeddings))
            return embeddings, ys, chans
        else:
            return None, None, None

    def load_model(self, checkpoint, resume=False):
        checkpoint_name = None
        if isinstance(checkpoint, str):
            checkpoint_name = checkpoint
            checkpoint = torch.load(checkpoint, map_location="cpu")
        assert_("model" in checkpoint, "Please provide correct checkpoint")
        model_dict = checkpoint["model"]
        try:
            self.net.load_state_dict(model_dict)
        except Exception as e:
            model_dict = {
                k.replace("module.", ""): v
                for k, v in model_dict.items()
                if k.startswith("module.")
            }
            self.net.load_state_dict(model_dict)

        if resume:
            if "optimizer" in checkpoint and "checkpoint_epoch" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                if checkpoint["checkpoint_epoch"] is not None:
                    self.start_epoch = checkpoint["checkpoint_epoch"] + 1
                    print(
                        ">>>> Resume checkpoint %s with optimizer"
                        % str(self.start_epoch)
                    )
                if "scaler" in checkpoint:
                    self.loss_scaler.load_state_dict(checkpoint["scaler"])
                    print(">>>> Resume checkpoint also with loss_scaler!")
        else:
            print(">>>> Successfullly load pre-trained checkpoint", checkpoint_name)


class Microsnoop(CoreModel):
    """
    Define Microsnoop Class
    """

    def __init__(self):
        super().__init__()

    def embed(
        self,
        local_rank,
        X,
        y,
        chan,
        args,
        model_type="cnn",
        tile=False,
        tile_overlap=0.1,
        tile_size=224,
        tile_rate=1.0,
        normalize=True,
    ):
        rank = args.node_rank * args.ngpu + local_rank
        dist.init_process_group(
            args.dist_backend,
            init_method="env://",
            rank=rank,
            world_size=args.world_size,
        )
        torch.cuda.set_device(local_rank)

        seed = args.seed + rank
        torch.manual_seed(seed)
        np.random.seed(seed)

        # print param
        global_rank = get_rank()
        nimg = len(X)
        nchan = X[0].shape[0]
        if global_rank == args.local_rank:
            print(
                f">>>> Input images with {nchan} channel, Batch_size: {args.batch_size}, Number of images: {nimg}"
            )

        embed_fn = None
        if model_type == "cnn":

            def embed_fn(latent):
                embedding = latent[-1]
                embedding = F.avg_pool2d(
                    embedding, kernel_size=(embedding.shape[-2], embedding.shape[-1])
                )
                flatten = nn.Flatten()
                embedding = flatten(embedding)
                embedding = (
                    embedding / torch.sum(embedding**2, axis=1, keepdim=True) ** 0.5
                )
                return embedding

        # set distributed model
        self.net.cuda(local_rank)
        find_unused_parameters = False
        if "uformer" in args.model_name:
            find_unused_parameters = True
        self.net = nn.parallel.DistributedDataParallel(
            self.net,
            device_ids=[local_rank],
            find_unused_parameters=find_unused_parameters,
        )
        # print('Rank state_dict()', local_rank, self.net.module.state_dict()['encoder.cls_token'])

        if not tile:
            embeddings, ys, chans = self._embed_imgs(
                X, y, chan, args, embed_fn=embed_fn, tile=tile
            )
            if global_rank == args.local_rank:
                embeddings, ys, chans = list(embeddings), list(ys), list(chans)
        else:
            embeddings, ys, chans = [], [], []
            for i in range(len(X)):
                Xi, ysub, xsub, Ly, Lx = make_tiles(
                    X[i], bsize=tile_size, augment=False, tile_overlap=tile_overlap
                )
                ny, nx, nchan, ly, lx = Xi.shape
                Xi = np.reshape(Xi, (ny * nx, nchan, ly, lx))
                if args.input_size != tile_size:
                    Xi = pad_image(
                        Xi, xy=[args.input_size, args.input_size]
                    )  # Note 1: resize到网络输入的大小
                Xi = np.array(Xi)[
                    np.random.choice(
                        range(len(Xi)), math.ceil(tile_rate * len(Xi)), replace=False
                    )
                ]
                if normalize:
                    Xi = np.array(
                        [normalize_img(Xi[j], axis=0) for j in range(len(Xi))]
                    )  # Note 2: 放在resize之后做归一化
                yi = np.expand_dims(y[i], 0).repeat(Xi.shape[0], axis=0)
                chani = np.expand_dims(chan[i], 0).repeat(Xi.shape[0], axis=0)
                embeddingsi, ysi, chansi = self._embed_imgs(
                    Xi, yi, chani, args, embed_fn=embed_fn, tile=tile
                )
                if global_rank == args.local_rank:
                    embeddings.append(embeddingsi)
                    ys.append(ysi)
                    chans.append(chansi)

        if global_rank == args.local_rank:
            if not os.path.isdir(args.embed_dir):
                os.makedirs(args.embed_dir)
            file_path = os.path.join(args.embed_dir, args.name_meta + ".h5")
            f_embedding = h5py.File(file_path, "w")
            # f_embedding = h5c.File(file_path, 'w', chunk_cache_mem_size=1024**3*10)
            f_embedding["embeddings"] = embeddings
            f_embedding["inds"] = ys
            f_embedding["chans"] = chans
            f_embedding.close()


class MicrosnoopCNN(Microsnoop):
    def __init__(
        self, model_init_param, checkpoint=None, input_size=224, patch_size=16
    ):
        super().__init__()
        # self.input_size = model_init_param['input_size']
        # self.patch_size = model_init_param['patch_size']
        self.input_size = input_size
        self.patch_size = patch_size
        self.net = CNNNet(**model_init_param)
        # self.net = CNNNet()
        if checkpoint is not None:
            self.load_model(checkpoint=checkpoint)

class MicrosnoopCustomWrapper(Microsnoop):
    def __init__(
        self, checkpoint=None, input_size=224, patch_size=16
    ):
        super().__init__()
        # self.input_size = model_init_param['input_size']
        # self.patch_size = model_init_param['patch_size']
        self.input_size = input_size
        self.patch_size = patch_size
        self.net = CNNNet()
        self.load_model(checkpoint="/path/to/microsnoop")
        self.parameters = self.net.parameters
        self.buffers = self.net.buffers

    def __call__(self, X):
        latent = self.net.encoder(X, mask_ratio=0)[0]
        return self.embed_fn(latent)

    def to(self, device):
        self.net.to(device)
        return self

    def embed_fn(self, latent):
            embedding = latent[-1]
            embedding = F.avg_pool2d(
                embedding, kernel_size=(embedding.shape[-2], embedding.shape[-1])
            )
            flatten = nn.Flatten()
            embedding = flatten(embedding)
            embedding = (
                embedding / torch.sum(embedding**2, axis=1, keepdim=True) ** 0.5
            )
            return embedding

class CNNNet(nn.Module):
    """
    refer to https://github.com/MouseLand/cellpose and https://github.com/cellimnet/scellseg-publish
    """

    def __init__(self, in_chans=1, out_chans=1, depths=[32, 64, 128, 256]):
        super().__init__()
        self.encoder = CNN_encoder(in_chans=in_chans, depths=depths)
        self.decoder = CNN_decoder(depths=depths, out_chans=out_chans)

    def forward(self, imgs, mask_ratio=None):
        latent, mask = self.encoder(imgs, mask_ratio)
        pred = self.decoder(latent)  # [N, L, p*p*3]
        return pred, mask




class CNN_encoder(nn.Module):
    def __init__(self, in_chans=1, depths=[32, 64, 128, 256], sz=3, residual_on=True):
        super().__init__()
        nbase = [in_chans] + depths
        self.downsample = CNN_downsample(nbase, sz=sz, residual_on=residual_on)

    def forward(
        self, x, mask_ratio=None
    ):  # 'mask_ratio': to harmonise with the vit model
        embeddings = self.downsample(x)
        mask = None  # to harmonise with the vit model
        return embeddings, mask




class CNN_decoder(nn.Module):
    def __init__(
        self,
        depths=[32, 64, 128, 256],
        out_chans=1,
        sz=3,
        residual_on=True,
        concatenation=False,
    ):
        super().__init__()
        self.depths = depths
        nbase = depths + [depths[-1]]
        self.upsample = CNN_upsample(
            nbase,
            sz=sz,
            residual_on=residual_on,
            concatenation=concatenation,
            style_channels=[depths[-1], depths[-1], depths[-1], depths[-1]],
        )
        self.make_style = makeStyle()
        self.base_bn = nn.BatchNorm2d(depths[0], eps=1e-5)
        self.base_relu = nn.ReLU(inplace=True)
        self.base_conv = nn.Conv2d(depths[0], out_chans, 1, padding=1 // 2)

    def forward(self, embeddings):
        styles = [self.make_style(embeddings[-1]) for _ in range(len(self.depths))]
        y = self.upsample(styles, embeddings)
        y = self.base_conv(self.base_relu(self.base_bn(y)))
        return y


class makeStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x0):
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True) ** 0.5
        return style


def normalize_img(img, axis=-1, invert=False):
    """
    optional inversion

    Parameters
    ------------
    img: ND-array (at least 3 dimensions)
    axis: channel axis to loop over for normalization

    Returns
    img: ndarray, float32
        normalized image of same size
    """
    if img.ndim < 3:
        raise ValueError("Image needs to have at least 3 dimensions")

    img = img.astype(np.float32)
    img = np.moveaxis(img, axis, 0)
    for k in range(img.shape[0]):
        if np.ptp(img[k]) > 0.0:
            img[k] = normalize99(img[k])
            if invert:
                img[k] = -1 * img[k] + 1
    img = np.moveaxis(img, 0, axis)
    return img


def normalize99(img):
    """normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile"""
    X = img.copy()
    if np.percentile(X, 99) - np.percentile(X, 1) == 0:
        X = (X - np.percentile(X, 1)) / (
            np.percentile(X, 99) - np.percentile(X, 1) + 1e-6
        )  # 这种归一化对0很多的图像不适用
    else:
        X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def pad_image(X, M=None, xy=None):
    """
    X: list or ndarray
    channel first
    """
    nimg = len(X)
    imgs = []

    if M is not None:
        masks = []
    for i in range(nimg):
        Ly, Lx = X[0].shape[-2:]
        dy = xy[0] - Ly
        dx = xy[1] - Lx
        ypad1, ypad2, xpad1, xpad2 = 0, 0, 0, 0
        if dy>0:
            ypad1 = int(dy // 2)
            ypad2 = dy - ypad1
        if dx>0:
            xpad1 = int((dx // 2))
            xpad2 = dx - xpad1

        if X[i].ndim == 3:
            nchan = X[0].shape[0]
            imgi = np.zeros((nchan, xy[0], xy[0]), np.float32)
            for m in range(nchan):
                imgi[m] = np.pad(X[i][m], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        elif X[i].ndim == 2:
            imgi = np.pad(X[i], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
        imgs.append(imgi)

        if M is not None:
            maski = np.pad(M[i], [[ypad1, ypad2], [xpad1, xpad2]], mode='constant')
            masks.append(maski)

# %%
