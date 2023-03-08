from typing import Union, Any, Optional, Callable
from decimal import Decimal

import torch
from torch import Tensor
from torchmetrics.image import FrechetInceptionDistance as FIDTorchmetrics
from gaussian_ot import sqrtm, make_psd, is_spd


def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    r"""Adjusted version of `Fid Score`_

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant - used if sigma_1 @ sigma_2 matrix is singular

    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2

    # Product might be almost singular
    if ~is_spd(sigma1, strict=True):
        sigma1, corr = make_psd(sigma1, strict=True, return_correction=True)
        print(f"Source covariance not positive definite. Adding {Decimal(corr.item()):2E} to the diagonal")
    if ~is_spd(sigma2, strict=True):
        sigma1, corr = make_psd(sigma2, strict=True, return_correction=True)
        print(f"Target covariance not positive definite. Adding {Decimal(corr.item()):2E} to the diagonal")

    tr_covmean = torch.trace(sqrtm(sigma2 @ sigma1))
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


class FrechetInceptionDistance(FIDTorchmetrics):
    def __init__(
        self,
        feature: Union[int, torch.nn.Module] = 2048,
        reset_real_features: bool = True,
        normalize: bool = True,
        preprocess: Optional[Callable[[Tensor], Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            feature=feature,
            reset_real_features=reset_real_features,
            normalize=normalize,
            **kwargs
        )
        self.preprocess = preprocess if preprocess is not None else torch.nn.Identity()

    def update(self, preds: Optional[Tensor] = None, target: Optional[Tensor] = None) -> None:  # type: ignore
        """ Update the state with extracted features
        Args:
            preds: tensor of ``fake`` images
            target: tensor of ``real`` images
        """

        if preds is not None: super().update(self.preprocess(preds), real=False)
        if target is not None: super().update(self.preprocess(target), real=True)
    #
    # def compute(self) -> Tensor:
    #     """Calculate FID score based on accumulated extracted features from the two distributions."""
    #     mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
    #     mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)
    #
    #     cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
    #     cov_real = cov_real_num / (self.real_features_num_samples - 1)
    #     cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
    #     cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
    #     return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)
