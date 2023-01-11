from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import (
    InterpolationMode,
    to_tensor,
    resize,
    rgb_to_grayscale,
)


SIZE = 270, 480

def region_of_interest(x):
    return x[..., 60:270, 60:440]


def transform(image, gray=False):
    image = np.flip(np.flipud(image), axis=2)
    image = image.copy()
    image = to_tensor(image)
    image = resize(image, SIZE)
    return image


def sort_key(x):
    return int(x[2:-4])


def load_psf(path):
    psf = np.array(Image.open(path))
    return transform(psf)


class LenslessLearning(Dataset):
    def __init__(self, diffuser_images, ground_truth_images):
        """
        Everything is upside-down, and the colors are BGR...
        """
        self.xs = diffuser_images
        self.ys = ground_truth_images

    def read_image(self, filename):
        image = np.load(filename)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        diffused = self.xs[idx]
        ground_truth = self.ys[idx]
        x = transform(np.load(diffused))
        y = transform(np.load(ground_truth))
        return x, y


class LenslessLearningInTheWild(Dataset):
    def __init__(self, path):
        xs = []
        manifest = sorted((x.name for x in path.glob('*.npy')))
        for filename in manifest:
            xs.append(path / filename)

        self.xs = xs

    def read_image(self, filename):
        image = np.load(filename)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        diffused = self.xs[idx]
        x = transform(np.load(diffused))
        return x


class LenslessLearningCollection:

    def __init__(self, path):
        path = Path(path)

        self.psf = load_psf(path / 'psf.tiff')

        train_diffused, train_ground_truth = load_manifest(path, 'dataset_train.csv')
        val_diffused, val_ground_truth = load_manifest(path, 'dataset_test.csv')

        self.train_dataset = LenslessLearning(train_diffused, train_ground_truth)
        self.val_dataset = LenslessLearning(val_diffused, val_ground_truth)
        self.region_of_interest = region_of_interest


def load_manifest(path, csv_filename):
    with open(path / csv_filename) as f:
        manifest = f.read().split()

    xs, ys = [], []
    for filename in manifest:
        x = path / 'diffuser_images' / filename.replace(".jpg.tiff", ".npy")
        y = path / 'ground_truth_lensed' / filename.replace(".jpg.tiff", ".npy")
        if x.exists() and y.exists():
            xs.append(x)
            ys.append(y)
        else:
            print(f"No file named {x}")
    return xs, ys
