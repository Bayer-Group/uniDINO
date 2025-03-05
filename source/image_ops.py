import numpy as np
import skimage
import torch
import tarfile
import io
from typing import Union
from PIL import Image
import tifffile
import json

MACHINE_EPS = np.finfo(np.float32).eps


def uint16_to_float(img):
    """Convert uint16 imags to float
    and convert to the range (0,1).
    Here: uint16_max = 2**16 - 1
    """
    img = img / (2**16 - 1)
    return img


def read_image_channels(filenames: Union[list, np.array]) -> np.array:
    """Read a multichannel image from a list of input files"""
    images = []
    for filename in filenames:
        images.append(skimage.io.imread(filename).astype(np.float32))
    return np.array(images)


def load_image_from_tar(tar, fname):
    """Function for loading images directly from tarred folder"""
    imgfile = tar.extractfile(fname).read()
    img = Image.open(io.BytesIO(imgfile))
    return np.array(img).astype(np.float32)


def read_image_from_tar(filenames, tarfname):
    images = []
    for filename in filenames:
        tar = tarfile.open(tarfname)
        img = load_image_from_tar(tar, fname=filename)
        images.append(img)
    return images


def scale_intensities(img: np.array, percentile: float = 99.9) -> np.array:
    """Scale intesities for each channel to the range [0,1] using the lower and higher percentiles to define the max and min values. Using percentiles provides robustness against fluorescence saturation artifacts.
    :param img: multichannel image of shape (C, W, H)
    :param percentile: percentile to define the lower and highest intensity values in the image. Value of 100 is equivalent to min-max scaling
    """
    if len(img.shape) == 2:
        axis = None
    else:
        axis = (1, 2)
    maxi = np.percentile(img, percentile, axis=axis, keepdims=True).astype(
        np.float32
    )  # no clue why output is float64
    mini = np.percentile(img, 100 - percentile, axis=axis, keepdims=True).astype(
        np.float32
    )
    # TODO for now adding a small constant in
    # the denominator to avoid division by zero
    img = (img - mini) / (MACHINE_EPS + maxi - mini)
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def generate_cellcrops(
    img: np.array,
    crop_size: int,
    stride: int,
    otsu_thresh_ratio: float = 0.7,
    otsu_down_factor: int = 10,
    otsu_chan: int = 0,
    min_area_ratio=0.01,
    otsuth=None,
) -> list:
    """
    Takes an image (C, W, H) as input and returns a list of crops of crop_size from the image.
    Difference to the other inference_cell_crop functions: returns overlapping crops.
    We assume quadratic images
    :param img: multi-channel image (C, W, H)
    :param crop_size: size of the crop in pixels
    #:ncrops_w: number of crops along the width dimension
    #:ncrops_h: number of crops along the height dimension
    :stride: stride between crops (think convolution operator)
    :returns: a list of crops each of dimension (C, crop_size)
    """

    # Get a binary mask based on the full image intensities
    img = torch.Tensor(img)
    if otsuth is None:
        thresh = pt_threshold_otsu(
            img[
                otsu_chan,
                0 : img.shape[1] : otsu_down_factor,
                0 : img.shape[2] : otsu_down_factor,
            ]
        )
    else:
        thresh = otsuth
    # has one dimension less now than img, i.e. (width, height)
    bin_img = img[otsu_chan, :, :] > otsu_thresh_ratio * thresh
    # bin_img = torch.Tensor(bin_img)
    # result is of shape: (ncrops_width, ncrops_height, crop_size, crop_size)
    bin_img_patches = bin_img.unfold(dimension=0, size=crop_size, step=stride).unfold(
        dimension=1, size=crop_size, step=stride
    )
    # result is of shape: (nchannels, ncrops_width, ncrops_height, crop_size, crop_size)
    img_patches = img.unfold(dimension=1, size=crop_size, step=stride).unfold(
        dimension=2, size=crop_size, step=stride
    )

    croplist = []
    labels = []
    crops_per_dim = img_patches.shape[1]
    for r in range(crops_per_dim):
        for c in range(crops_per_dim):
            crop = img_patches[:, r, c, ...]
            bin_crop = bin_img_patches[r, c, ...]
            cell_area_ratio = bin_crop.sum() / (bin_crop.shape[0] * bin_crop.shape[1])
            enough_cells = cell_area_ratio >= min_area_ratio
            croplist.append(crop)
            labels.append(enough_cells)

    if np.sum(labels) == 0:
        labels[int(len(labels) / 2)] = True

    return croplist, labels


def pt_histogram(image, nbins=256, normalize=False):
    """(Torch implementation) Return histogram of image.
    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution.
    If `channel_axis` is not set, the histogram is computed on the flattened
    image. For color or multichannel images, set ``channel_axis`` to use a
    common binning for all channels. Alternatively, one may apply the function
    separately on each channel to obtain a histogram for each color channel
    with separate binning.
    Parameters"""

    image = torch.flatten(image)
    # For integer types, histogramming with bincount is more efficient.
    hist = torch.histc(image, bins=nbins)
    device = torch.device("cuda" if hist.is_cuda else "cpu")
    bin_edges = torch.linspace(
        torch.min(image), torch.max(image), steps=nbins + 1, device=device
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    if normalize:
        hist = hist / torch.sum(hist)

    return hist, bin_centers


def pt_threshold_otsu(image=None, nbins=256):
    """(Torch implementation) Return threshold value based on Otsu's method.
    Either image or hist must be provided. If hist is provided, the actual
    histogram of the image is ignored. The input image must be grayscale.
    """
    # Check if the image has more than one intensity value; if not, return that
    # value
    if image is not None:
        first_pixel = torch.ravel(image)[0]
        if torch.all(image == first_pixel):
            return first_pixel

    counts, bin_centers = pt_histogram(image, nbins)

    # class probabilities for all possible thresholds
    weight1 = torch.cumsum(counts, dim=0)
    weight2 = torch.flip(torch.cumsum(torch.flip(counts, dims=(0,)), dim=0), dims=(0,))
    # class means for all possible thresholds
    mean1 = torch.cumsum(counts * bin_centers, dim=0) / weight1
    mean2 = torch.cumsum(
        torch.flip(counts * bin_centers, dims=(0,)), dim=0
    ) / torch.flip(weight2, dims=(0,))
    mean2 = torch.flip(mean2, dims=(0,))
    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = torch.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold


def load_tiff_img(path, channel=None, include_metadata=False):
    with tifffile.TiffFile(path) as tif:
        if channel is not None:
            img = tif.series[0][channel]
        else:
            img = tif.series[0]
        img = img.asarray().astype(np.float32)
        metadata = None
        if include_metadata:
            metadata = tif.pages[0].tags["ImageDescription"].value
            metadata = json.loads(metadata)
    return img, metadata


def load_np_img(path):
    return np.load(path, mmap_mode="r")
