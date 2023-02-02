"""
************************************************************************************************************************

`PyTorch <https://pytorch.org/>`_ implementation of matrix utilities

Implemented by: `Theo J. Adrai <https://github.com/theoad>`_ All rights reserved

.. warning:: Work in progress. This implementation is still being verified.

.. _TheoA: https://github.com/theoad

************************************************************************************************************************
"""
from typing import Union, Tuple

import torch
from torch.distributions import MultivariateNormal
from torch import Tensor, BoolTensor
from tqdm import tqdm


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


@torch.inference_mode()
def get_statistics(model, dl, ddp, desc, size, normalize=None, mmse=None):
    sum = torch.zeros(size, dtype=torch.double, device=ddp.device)
    sum_cov = torch.zeros(size, size, dtype=torch.double, device=ddp.device)
    n = torch.zeros(1, dtype=torch.long, device=ddp.device)

    for x, _ in tqdm(dl, desc=desc, disable=not ddp.is_local_main_process):
        if mmse is not None: x = mmse(x)
        if normalize is not None: x = normalize(x)
        latent = model(x).z
        latent = latent.permute(0, 2, 3, 1).flatten(0, -2).double()
        n += latent.size(0)
        sum += latent.sum(0)
        sum_cov += torch.einsum("bi,bj->ij", latent, latent)

    # Reduce across devices
    ddp.wait_for_everyone()
    n = ddp.reduce(n)
    mu = ddp.reduce(sum) / n
    cov = (ddp.reduce(sum_cov) - n * mu[:, None] @ mu[None, :]) / (n - 1)
    return mu, cov

@torch.inference_mode()
def get_transport(Cs, Ct, mus, mut, ddp, model):
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

    ddp.wait_for_everyone()
    def transport(latents, pg_star: float = 0):
        b,c,h,w = latents.shape
        latent = latents.permute(0,2,3,1).flatten(0, -2).double()
        T = (1 - pg_star) * T0 + pg_star * eye_like(T0)
        transported = (T @ (latent - mus)[..., None]).squeeze() + mut
        if Cw0 is not None:
            transported += MultivariateNormal(torch.zeros_like(Cw0[0]), Cw0 * (1 - pg_star) ** 0.5).sample((latent.size(0),))
        transported = transported.float().unflatten(0, [b,h,w]).permute(0,3,1,2).contiguous()
        return ddp.unwrap_model(model).decode(transported).sample
    return transport
