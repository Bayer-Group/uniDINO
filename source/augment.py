import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from source.utils import assert_


class RandomCropWithCells(nn.Module):
    """Transform class that returns a random crop
    with at least 1% of area covered with cells
    """

    def __init__(
        self,
        size,
        min_area_ratio: float = 0.01,
        otsu_chan: int = 4,
        otsu_down_factor: int = 10,
        otsu_thresh_ratio: float = 0.7,
        max_try: int = 20,
        scale: tuple = None,
    ):
        super().__init__()
        assert_(isinstance(size, (int, tuple)))
        self.sample_crop = transforms.RandomCrop(size=size)
        self.size = size
        if scale:
            self.sample_crop = transforms.RandomResizedCrop(size=size, scale=scale)
        self.min_area_ratio = min_area_ratio
        self.otsu_chan = otsu_chan
        self.otsu_down_factor = otsu_down_factor
        self.otsu_thresh_ratio = otsu_thresh_ratio
        self.max_try = max_try

    def forward(self, img, thresh=None):
        # We do not use threshold here

        img = torch.Tensor(img)
        img_patch = self.sample_crop(img)

        return img_patch


class RandomDropChannel(nn.Module):
    """Transform class that drops a channel at random"""

    def __init__(self):
        super().__init__()

    def forward(self, img):
        # extract number of channels
        nchan, _, _ = img.shape
        # pick a channel at random
        ch = np.random.randint(low=0, high=nchan)
        img[ch] = 0
        return img


class GaussianNoise(nn.Module):

    def __init__(self, sigma=0.05):
        super().__init__()
        self.sigma = sigma

    def forward(self, img):
        # add Gaussian noise
        img_with_noise = img + self.sigma * torch.randn_like(img)
        return img_with_noise


class RandomChannelShift(nn.Module):
    """Transform class that shifts each channel at random"""

    def __init__(self, min=-0.1, max=0.1):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, img):
        # exract number of channels
        nchan, _, _ = img.shape
        for c in range(nchan):
            shift = (self.max - self.min) * torch.rand(
                1,
            ) + self.min
            img[c] = img[c] + shift
        img[img > 1] = 1
        img[img < 0] = 0
        return img


class RandomBrightnessChange(nn.Module):
    """Transform class that changes brightness of color channels at random"""

    def __init__(self, min=0.8, max=1.2):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, img):
        # exract number of channels
        nchan, _, _ = img.shape
        for c in range(nchan):
            gamma = (self.max - self.min) * torch.rand(
                1,
            ) + self.min
            img[c] = img[c] ** gamma
        img[img > 1] = 1
        img[img < 0] = 0
        return img


class ContrastiveTransformations:
    """Returns n_views (augmentations) of the same image"""

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


class Augment:
    """Handles the augmentation logic"""

    def __init__(self, crop_transform, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.crop_transform = crop_transform
        self.n_views = n_views

    def __call__(self, x, metadata):
        if isinstance(metadata, dict):
            th = float(metadata["otsuth"])
        else:
            th = metadata

        views = []
        for _ in range(self.n_views):
            views.append(self.base_transforms(self.crop_transform(x, th)))
        return views


def get_default_augs(
    color_prob=1,
    means=None,
    sds=None,
):
    """Default cell-painting augmentation pipeline based on the ablation study.
    Crop and multiple views are NOT included.
    """
    min_shift, max_shift = (-0.3, 0.3)
    min_bright, max_bright = (0.5, 1.5)
    drop_prob = 0
    noise_prob = 0
    blur_prob = 0
    kernel_size = 23
    means = torch.tensor(means)
    sds = torch.tensor(sds)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply(
                [
                    RandomChannelShift(min=min_shift, max=max_shift),
                    RandomBrightnessChange(min=min_bright, max=max_bright),
                ],
                p=color_prob,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=kernel_size)], p=noise_prob
            ),
            transforms.RandomApply([GaussianNoise()], p=blur_prob),
            transforms.RandomApply([RandomDropChannel()], p=drop_prob),
            transforms.Normalize(means, sds),
        ]
    )
    return transform
