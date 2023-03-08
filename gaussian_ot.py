"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of matrix utilities

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Tuple, Union, Any, Optional, Literal
from collections import namedtuple
from functools import partial
from copy import deepcopy

import torch
from torch.distributions import MultivariateNormal
from torch import Tensor, BoolTensor
from torchmetrics import Metric
from einops import rearrange
import utils


STABILITY_CONST = 1e-8


# Function from the pyRiemann package ported in pytorch
def _matrix_operator(matrices: Tensor, operator) -> Tensor:
    """
    Matrix equivalent of an operator. Works batch-wise
    Porting of pyRiemann to pyTorch
    Original Author: Alexandre Barachant
    https://github.com/alexandrebarachant/pyRiemann
    """
    eigvals, eigvects = torch.linalg.eigh(matrices, UPLO='L')
    eigvals = torch.diag_embed(operator(eigvals))
    return eigvects @ eigvals @ eigvects.transpose(-2, -1)


def eye_like(matrices: Tensor) -> Tensor:
    """
    Return Identity matrix with the same shape, device and dtype as matrices

    :param matrices: Batch of matrices with shape [*, C, D] where * is zero or leading batch dimensions
    :return: Tensor T with shape [*, C, D]. with T[i] = torch.eye(C, D)
    """
    return torch.eye(*matrices.shape[-2:-1], out=torch.empty_like(matrices)).expand_as(matrices)


def sqrtm(matrices: Tensor) -> Tensor:
    """
    :param matrices: batch of SPSD matrices
    :returns: batch containing mat. square root of each matrix
    """
    return _matrix_operator(matrices, torch.sqrt)


def invsqrtm(matrices: Tensor) -> Tensor:
    """
    :param matrices: batch of SPD matrices
    :returns: batch containing mat. inverse sqrt. of each matrix
    """
    isqrt = lambda x: 1. / torch.sqrt(x)
    return _matrix_operator(matrices, isqrt)


def is_symmetric(matrices: Tensor) -> BoolTensor:
    """
    Boolean method. Checks if matrix is symmetric.

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :return: Boolean tensor T with shape [*]. with T[i] == True <=> matrices[i] is symmetric
    """
    if matrices.size(-1) != matrices.size(-2):
        return torch.full_like(matrices.mean(dim=(-1, -2)), 0).bool()  # = Tensor([False, False, ..., False])
    return torch.sum((matrices - matrices.transpose(-2, -1))**2, dim=(-1, -2)) < STABILITY_CONST  # noqa


def min_eig(matrices: Tensor) -> Tensor:
    """
    Returns the minimal eigen values of a batch of matrices (signed).

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :return: Tensor T with shape [*]. with T[i] = min(eig(matrices[i]))
    """
    return torch.min(torch.linalg.eigh(matrices)[0], dim=-1)[0]


def is_pd(matrices: Tensor, strict=True) -> BoolTensor:
    """
    Boolean method. Checks if matrices are Positive Definite (PD).

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :param strict: If ``False`` checks the matrices are positive semi-definite
    :return: Boolean tensor T with shape [*]. with T[i] == True <=> matrices[i] is PD
    """
    return min_eig(matrices) > 0 if strict else min_eig(matrices) >= 0


def is_spd(matrices: Tensor, strict=True) -> BoolTensor:
    """
    Boolean method. Checks if matrices are Symmetric and Positive Definite (SPD).

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :param strict: If ``False`` checks the matrices are positive semi-definite (SPSD)
    :return: Boolean tensor T with shape [*]. with T[i] == True <=> matrices[i] is SPD
    """
    return torch.logical_and(is_symmetric(matrices), is_pd(matrices, strict=strict)).bool()


def make_psd(matrices: Tensor, strict: bool = False, return_correction: bool = False, diag: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Add to each matrix its minimal eigen value to make it positive definite.

    :param matrices: Batch of matrices with shape [*, D, D] where * is zero or leading batch dimensions
    :param strict: If ``True``, add a small stability constant to make the matrices positive definite (PD)
    :param return_correction: If ``True``, returns the correction added to the diagonal of the matrices.
    :return: Tensor T with shape [*]. with T[i] = matrices[i] + min(eig(matrices[i]) * I
    """
    smallest_eig = matrices.min(-1)[0] if diag else min_eig(matrices)
    small_positive_val = smallest_eig.clamp(max=0).abs()
    if strict: small_positive_val += STABILITY_CONST
    if diag:
        res = matrices + small_positive_val[..., None]
    else:
        I = eye_like(matrices)
        res = matrices + I * small_positive_val[..., None, None]
    if return_correction:
        return res, small_positive_val
    return res


def w2_gaussian(
        mean_source: Tensor,
        mean_target: Tensor,
        cov_source: Tensor,
        cov_target: Tensor,
) -> Tensor:
    """
    Computes closed form squared W2 distance between Gaussian distribution_models (also known as Gelbrich Distance)
    :param mean_source: A 1-dim vectors representing the source distribution mean with optional leading batch dims [*, D]
    :param mean_target: A 1-dim vectors representing the target distribution mean with optional leading batch dims [*, D]
    :param cov_source: A 2-dim matrix representing the source distribution covariance [*, D, D]
    :param cov_target: A 2-dim matrix representing the target distribution covariance [*, D, D]

    :return: The squared Wasserstein 2 distance between N(mean_source, cov_source) and N(mean_target, cov_target)
    """
    cov_target_sqrt = sqrtm(cov_target)
    mix = cov_target_sqrt @ cov_source @ cov_target_sqrt
    mean_shift = torch.sum((mean_source - mean_target) ** 2, dim=-1)
    cov_shift_trace = torch.diagonal(cov_source + cov_target - 2 * sqrtm(mix), dim1=-2, dim2=-1).sum(dim=-1)
    return mean_shift + cov_shift_trace


def gaussian_transport_operators(Cs, Ct):
    I = eye_like(Cs)
    if is_spd(Cs, strict=True):
        sqrtCs, isqrtCs = sqrtm(Cs), invsqrtm(make_psd(Cs, strict=True))
        T0 = (isqrtCs @ sqrtm(make_psd(sqrtCs @ Ct @ sqrtCs, strict=True)) @ isqrtCs)
        Cw0 = None
    else:
        sqrtCt, isqrtCt = sqrtm(Ct), invsqrtm(make_psd(Ct, strict=True))
        mix = sqrtm(make_psd(sqrtCt @ Cs @ sqrtCt, strict=True))
        T_star = (isqrtCt @ mix @ isqrtCt)
        pinvCs = torch.linalg.pinv(Cs)
        T0 = (sqrtCt @ mix @ isqrtCt @ pinvCs)
        Cw0 = sqrtCt @ (I - sqrtCt @ T_star @ pinvCs @ T_star @ sqrtCt) @ sqrtCt

    return T0, Cw0


Reshape = namedtuple('Reshape', 'dim rearrange rearrange_back')


class GaussianOT(Metric):

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    source_features_sum: Tensor
    source_features_cov_sum: Tensor
    source_features_num_samples: Tensor

    target_features_sum: Tensor
    target_features_cov_sum: Tensor
    target_features_num_samples: Tensor

    mean_source: Optional[Tensor]
    mean_target: Optional[Tensor]
    transport_operator: Optional[Tensor]
    noise_covariance: Optional[Tensor]

    def __init__(
            self,
            latent_size: utils.LatentSize,
            autoencoder: torch.nn.Module,
            embed: Literal["pixel", "channel", "image"] = "image",
            reset_target_features: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.autoencoder = autoencoder

        latent_reshape = {
            "pixel": Reshape(
                dim=latent_size.c,
                rearrange=partial(rearrange, pattern='b c h w -> (b h w) c'),
                rearrange_back=partial(rearrange, pattern='(b h w) c -> b c h w', **latent_size._asdict())
            ),
            "channel": Reshape(
                dim=latent_size.h * latent_size.w,
                rearrange=partial(rearrange, pattern='b c h w -> (b c) (h w)'),
                rearrange_back=partial(rearrange, pattern='(b c) (h w) -> b c h w', **latent_size._asdict())
            ),
            "image": Reshape(
                dim=latent_size.c * latent_size.h * latent_size.w,
                rearrange=partial(rearrange, pattern='b c h w -> b (c h w)'),
                rearrange_back=partial(rearrange, pattern='b (c h w) -> b c h w', **latent_size._asdict()),
            ),
        }
        self.embed = latent_reshape[embed]

        if not isinstance(reset_target_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_target_features = reset_target_features

        mx_nb_feets = (self.embed.dim, self.embed.dim)
        self.add_state("source_features_sum", torch.zeros(self.embed.dim).double(), dist_reduce_fx="sum")
        self.add_state("source_features_cov_sum", torch.zeros(mx_nb_feets).double(), dist_reduce_fx="sum")
        self.add_state("source_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("target_features_sum", torch.zeros(self.embed.dim).double(), dist_reduce_fx="sum")
        self.add_state("target_features_cov_sum", torch.zeros(mx_nb_feets).double(), dist_reduce_fx="sum")
        self.add_state("target_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.register_buffer("transport_operator", None)
        self.register_buffer("noise_covariance", None)
        self.register_buffer("mean_source", None)
        self.register_buffer("mean_target", None)

    def update(self, source: Optional[Tensor] = None, target: Optional[Tensor] = None) -> None:  # type: ignore
        if source is not None:
            features = self.autoencoder.encode(source).double()
            features = self.embed.rearrange(features)

            self.source_features_sum += features.sum(dim=0)
            self.source_features_cov_sum += features.t().mm(features)
            self.source_features_num_samples += features.shape[0]

        if target is not None:
            features = self.autoencoder.encode(target).double()
            features = self.embed.rearrange(features)

            self.target_features_sum += features.sum(dim=0)
            self.target_features_cov_sum += features.t().mm(features)
            self.target_features_num_samples += features.shape[0]

    def transport_features(self, features: Tensor, pg_star: float = 0.) -> Tensor:
        T0, Cw0 = self.transport_operator, self.noise_covariance
        if T0 is None:
            raise NotImplementedError(
                "transport operator is not defined. You should call `compute` before calling `transport`."
            )

        T = (1 - pg_star) * T0 + pg_star * eye_like(T0) if pg_star > 0 else T0
        features_centered = (features - self.mean_source)
        transported = (T @ features_centered[..., None]).squeeze()
        if pg_star < 0:
            direction = transported - features_centered
            transported -= pg_star * direction
        transported += self.mean_target
        if Cw0 is not None and pg_star != 1:
            transported += MultivariateNormal(torch.zeros_like(Cw0[0]), Cw0 * (1 - pg_star) ** 0.5).sample(
                (features.size(0),))
        return transported

    def transport(self, imgs: Tensor, pg_star: float = 0.) -> Tensor:
        if hasattr(self, 'pg_star'): pg_star = self.pg_star
        features = self.autoencoder.encode(imgs)
        orig_type = features.dtype

        features = self.embed.rearrange(features).double()
        transported = self.transport_features(features, pg_star=pg_star)
        transported = self.embed.rearrange_back(transported).to(orig_type)

        transported_images = self.autoencoder.decode(transported)
        return transported_images

    def compute(self) -> Tensor:
        self.mean_source = self.source_features_sum / self.source_features_num_samples
        self.mean_target = self.target_features_sum / self.target_features_num_samples

        cov_source_num = self.source_features_cov_sum - self.source_features_num_samples * self.mean_source.unsqueeze(1).mm(self.mean_source.unsqueeze(0))
        cov_source = cov_source_num / (self.source_features_num_samples - 1)
        cov_target_num = self.target_features_cov_sum - self.target_features_num_samples * self.mean_target.unsqueeze(1).mm(self.mean_target.unsqueeze(0))
        cov_target = cov_target_num / (self.target_features_num_samples - 1)

        self.transport_operator, self.noise_covariance = gaussian_transport_operators(cov_source, cov_target)
        return w2_gaussian(self.mean_source, self.mean_target, cov_source, cov_target)

    def reset(self) -> None:
        if not self.reset_target_features:
            target_features_sum = deepcopy(self.target_features_sum)
            target_features_cov_sum = deepcopy(self.target_features_cov_sum)
            target_features_num_samples = deepcopy(self.target_features_num_samples)
            super().reset()
            self.target_features_sum = target_features_sum
            self.target_features_cov_sum = target_features_cov_sum
            self.target_features_num_samples = target_features_num_samples
        else:
            super().reset()

        self.mean_source = None
        self.mean_target = None
        self.transport_operator = None
        self.noise_covariance = None

    def forward(self, x):
        return self.transport(x)
