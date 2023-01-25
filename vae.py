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
    def __init__(self, in_channels: int = 3, *args, normalization = ((*((0.5,) * 3), 0), (*((0.5,) * 3), 1)), **kwargs):
        super().__init__(in_channels, *args, **kwargs)
        # take care of normalization within class
        self.normalization = tuple(map(lambda t: t[:in_channels], normalization))

    def norm(self, images):
        if self.normalization is None:
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    def forward(    # noqa
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        recon_loss: str = "mse_loss",
        beta: float = 1.
    ) -> Union[VAEOutput, torch.FloatTensor]:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        dec = self.decode(z).sample
        if not return_dict:
            return dec

        D = getattr(F, recon_loss)
        recon_loss = D(dec, sample)
        kl_unscaled = posterior.kl().mean() / sample[0].numel()
        kl = kl_unscaled * beta
        loss = recon_loss + kl
        return VAEOutput(
            posterior=posterior,
            z=z, recon=dec,
            kl=kl, kl_unscaled=kl_unscaled,
            recon_loss=recon_loss,
            total_loss=loss
        )
