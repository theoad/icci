import torch
import torch.nn.functional as F
from typing import Optional, Union
from diffusers.models.vae import AutoencoderKL, BaseOutput, DiagonalGaussianDistribution
from einops import rearrange


class VAEOutput(BaseOutput):
    posterior: DiagonalGaussianDistribution
    z: torch.FloatTensor
    recon: torch.FloatTensor
    kl: torch.FloatTensor
    kl_unscaled: torch.FloatTensor
    recon_loss: torch.FloatTensor
    total_loss: torch.FloatTensor


class VAE(AutoencoderKL):
    def __init__(self, in_channels: int = 3, sample_posterior: Optional[bool] = True, *args, **kwargs):
        super().__init__(in_channels, *args, **kwargs)
        self.sample_posterior = sample_posterior

    def forward(    # noqa
        self,
        sample: torch.FloatTensor,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        recon_loss: str = "mse_loss",
        beta: float = 1.
    ) -> Union[VAEOutput, torch.FloatTensor]:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if self.sample_posterior else posterior.mode()
        dec = self.decode(z).sample
        if not return_dict:
            return dec

        Distortion = getattr(F, recon_loss)
        recon_loss = Distortion(dec, sample)
        kl_unscaled = posterior.kl().mean() / sample[0].numel()
        kl = kl_unscaled * beta
        loss = recon_loss + kl if beta > 0 else recon_loss
        return VAEOutput(
            sample=sample,
            posterior=posterior,
            z=z,
            recon=dec,
            kl=kl,
            kl_unscaled=kl_unscaled,
            recon_loss=recon_loss,
            total_loss=loss
        )
