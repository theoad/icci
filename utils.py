from typing import Union, Sequence, Tuple, List
from itertools import accumulate
from collections import namedtuple
import numpy as np
import re
import math

import torch
from torchvision.transforms import Resize, GaussianBlur, RandomErasing, RandomApply
from torch.utils.data import Dataset, Subset

from diffusers.utils import BaseOutput

LatentSize = namedtuple('LatentSize', 'c h w')

class Downsample(torch.nn.Sequential):
    def __init__(self, size: int, scale_factor: int):
        super().__init__(Resize((size//scale_factor,) * 2), Resize((size,) * 2))

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma / 255.

    def forward(self, x):
        return x + torch.randn_like(x) * self.sigma

class Blur(GaussianBlur):
    def __init__(self, kernel_size: int, sigma: Tuple[int, int]):
        super(Blur, self).__init__(kernel_size=kernel_size, sigma=sigma)

class RandomMask(RandomErasing):
    def __init__(self):
        super().__init__(p=1., scale=list(np.linspace(0.1, 2, 10)), ratio=list(np.linspace(0.01, 5, 10)))

class Blind(torch.nn.Sequential):
    def __init__(self):
        super(Blind, self).__init__(
            Downsample(512, 8),
            AddGaussianNoise(50.),
        )

class CosineAnnealing(object):
    def __init__(self, n_step, min_val, max_val, decreasing=False):
        self._step = 0
        self.n_step = n_step
        self.min_val = min_val
        self.max_val = max_val
        self.decreasing = decreasing

    def step(self):
        self._step += 1

    @property
    def val(self):
        return self.cosine_annealing(self._step, self.n_step, self.min_val, self.max_val, self.decreasing)

    @staticmethod
    def cosine_annealing(step, n_step, min_val, max_val, decreasing=False):
        if n_step is None or step > n_step: return max_val
        return min_val + (max_val-min_val) * (0.5 * math.cos(math.pi * (step / n_step + int(not decreasing))) + 0.5)


def random_split(*dataset: Dataset, split: Sequence[Union[int, float]], seed=0) -> Tuple[Subset, ...]:
    length = len(dataset[0])  # noqa
    assert all(len(d) == length for d in dataset)

    if not all(isinstance(s, int) for s in split):
        assert all(0 < s < 1 for s in split) and sum(split) == 1
        split = list(map(lambda frac: int(frac * length), split))
        split[-1] = length - sum(split[:-1])

    if len(dataset) > 1: assert len(split) == len(dataset)
    assert sum(split) == length

    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    if len(dataset) > 1:
        return tuple(Subset(d, indices[offset - length: offset]) for d, offset, length in zip(dataset, accumulate(split), split))
    else:
        return tuple(Subset(dataset[0], indices[offset - length: offset]) for offset, length in zip(accumulate(split), split))


def unpatchify(patchified_pixel_values, patch_size, num_channels):
    num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)
    # sanity check
    if num_patches_one_direction ** 2 != patchified_pixel_values.shape[1]:
        raise ValueError("Make sure that the number of patches can be squared")

    # unpatchify
    batch_size = patchified_pixel_values.shape[0]
    patchified_pixel_values = patchified_pixel_values.reshape(
        batch_size,
        num_patches_one_direction,
        num_patches_one_direction,
        patch_size,
        patch_size,
        num_channels,
    )
    patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
    pixel_values = patchified_pixel_values.reshape(
        batch_size,
        num_channels,
        num_patches_one_direction * patch_size,
        num_patches_one_direction * patch_size,
    )
    return pixel_values


def camel2snake(name, sep='_'):
    return name[0].lower() + re.sub(r'(?!^)[A-Z]', lambda x: sep + x.group(0).lower(), name[1:])

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
