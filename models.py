import math
from typing import Optional, Tuple, Union
import numpy as np
from sympy import divisors

import torch
import torch.nn.functional as F
import tv_forward
from types import MethodType
from typing import Literal
import torchvision.models
from torchvision.transforms import ToPILImage, Compose, Normalize, Resize, ToTensor
from diffusers import UNet2DModel, ModelMixin, ConfigMixin, StableDiffusionPipeline
from diffusers.configuration_utils import register_to_config
from diffusers.models.vae import Decoder, Encoder
import utils


ModelName = Literal["ResNet34", "ViT_B_16", "MobileNet_V2", "EfficientNet_V2_S"]


class AEOutput(utils.BaseOutput):
    loss: torch.FloatTensor
    latent: torch.FloatTensor
    preds: torch.FloatTensor
    target: torch.FloatTensor


class AutoEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            name: ModelName,
            loss: str = "mse_loss"
    ) -> None:
        super().__init__()
        # Extract model from torchvision hub
        get_model = getattr(torchvision.models, name.lower())
        no_pooling_forward = getattr(tv_forward, name.lower() + '_forward')

        weights = getattr(torchvision.models, name + '_Weights').DEFAULT
        preprocess = weights.transforms()
        mean, std = preprocess.mean, preprocess.std
        sample_size = preprocess.crop_size[0]
        preprocess.mean, preprocess.std = [0], [1]
        encoder = get_model(weights=weights)
        encoder.forward = MethodType(no_pooling_forward, encoder)  # replace forward function to remove pooling
        encoder = tv_forward.remove_unused_params(encoder)  # remove classifier heads and pooling to avoid ddp_unused_parameters
        encoder = encoder.eval().requires_grad_(False)

        def no_train(self, mode=False):
            pass

        encoder.train = MethodType(no_train, encoder)  # keep the model in eval always

        dummy_img = ToPILImage()(torch.randn(3, 500, 400))
        x = preprocess(dummy_img).unsqueeze(0)
        latent_channels, latent_resolution, latent_resolution = encoder(x).shape[1:]

        num_layers = int(math.log2(sample_size // latent_resolution)) + 1
        norm_num_groups = div_sqrt(latent_channels)
        capacity = norm_num_groups * 2 ** (num_layers + 1)
        layers = tuple(min(max(capacity // 2 ** i, 2 * norm_num_groups), latent_channels) for i in range(1, num_layers + 1))

        self.encoder = encoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=3,
            up_block_types=("UpDecoderBlock2D",) * num_layers,
            norm_num_groups=norm_num_groups,
            layers_per_block=1,
            block_out_channels=layers,
        )

        self.sample_size = sample_size
        self.latent_size = utils.LatentSize(c=latent_channels, h=latent_resolution, w=latent_resolution)

        self.preprocess = preprocess
        self.normalize = Normalize(mean, std)
        self.denormalize = torch.nn.Sequential(
            Normalize(torch.as_tensor(mean).zero_(), 1. / torch.as_tensor(std)),
            Normalize(-torch.as_tensor(mean), torch.as_tensor(std).zero_() + 1.)
        )

        self.loss = getattr(F, loss)

    def encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.encoder(x)

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        return self.decoder(z)

    def forward(self, pixel_values: torch.FloatTensor) -> AEOutput:
        if self.encoder_training:
            latent = self.encode(pixel_values)
        else:
            # Without this, DDP screams that "your module has parameters that
            # were not used in producing loss" blabla....
            with torch.no_grad():
                latent = self.encode(pixel_values)

        preds = self.decode(latent)
        return AEOutput(loss=self.loss(preds, pixel_values), preds=preds, latent=latent, target=pixel_values)


class MMSEOutput(utils.BaseOutput):
    preds: torch.FloatTensor
    target: torch.FloatTensor
    loss: torch.FloatTensor


class MMSE(UNet2DModel):
    def forward(
            self,
            degraded: torch.FloatTensor,
            pixel_values: Optional[torch.FloatTensor] = None,
            return_dict: bool = True
    ) -> Union[MMSEOutput, Tuple, torch.Tensor]:
        preds = super().forward(degraded, 0.).sample
        if pixel_values is None: return preds
        if not return_dict: return preds,
        return MMSEOutput(preds=preds, target=pixel_values, loss=F.mse_loss(preds, pixel_values))


class SDAutoEncoder(ModelMixin, ConfigMixin):
    def __init__(self, sample_size = None):
        super().__init__()
        # Extract VAE from stable diffusion
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir=".cache/")
        self.model = pipe.vae.eval().requires_grad_(False)

        self.sample_size = sample_size or self.model.config.sample_size
        self.latent_channels = self.model.config.latent_channels
        self.latent_resolution = self.sample_size // pipe.vae_scale_factor

        mean, std = (0.5,) * 3, (0.5,) * 3
        self.preprocess = Compose([Resize((self.sample_size,) * 2), ToTensor()])

        self.normalize = Normalize(mean, std)
        self.denormalize = torch.nn.Sequential(
            Normalize(torch.as_tensor(mean).zero_(), 1./torch.as_tensor(std)),
            Normalize(-torch.as_tensor(mean), torch.as_tensor(std).zero_() + 1.)
        )
        self.latent_size = utils.LatentSize(c=self.latent_channels, h=self.latent_resolution, w=self.latent_resolution)

    def encode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.model.encode(x).latent_dist.sample()

    def decode(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.model.decode(x).sample


def div_sqrt(n: int) -> int:
    """
    Return `n` divisor that is closest to sqrt(n).

    :param n: positive integer
    :return: divisor of n that is the closest to sqrt(n)
    """
    assert isinstance(n, int) and n > 0, f"Error, n must be a positive integer. Given n={n}."
    divs = np.array(divisors(n))
    if len(divs) < 1: return 1
    sqrt_idx = np.searchsorted(divs, n ** 0.5)
    div = divs[sqrt_idx]
    return div
