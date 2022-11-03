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
    def __init__(self, path, start, end):
        """
        Everything is upside-down, and the colors are BGR...
        """
        diffuser_path = path / 'diffuser_images'
        ground_truth_path = path / 'ground_truth_lensed'

        xs, ys = [], []
        manifest = sorted((x.name for x in diffuser_path.glob('*.npy')), key=sort_key)
        for filename in manifest:
            xs.append(diffuser_path / filename)
            ys.append(ground_truth_path / filename)

        self.xs = xs
        self.ys = ys
        self.start, self.end = start, end

    def read_image(self, filename):
        image = np.load(filename)

    def __len__(self):
        return len(self.xs[self.start:self.end])

    def __getitem__(self, idx):
        diffused = self.xs[self.start + idx]
        ground_truth = self.ys[self.start + idx]
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
        self.train_dataset = LenslessLearning(path, start=1000, end=10000)
        self.val_dataset = LenslessLearning(path, start=0, end=1000)
        self.region_of_interest = region_of_interest
