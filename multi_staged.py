import torch
import torch.nn as nn
import utils
import vgg_new
import torchvision.transforms as T

class AutoEncoder(nn.Module):
    def __init__(self, index: int):
        super().__init__()
        sd = torch.load(f'convert_torch_to_pytorch/vgg_normalised_conv{index}_1.pth')
        self.encoder = getattr(vgg_new, 'encoder'+str(index))(sd)
        sd = torch.load(f'convert_torch_to_pytorch/feature_invertor_conv{index}_1.pth')
        self.decoder = getattr(vgg_new, 'decoder'+str(index))(sd)
        self.sample_size = 224
        dummy = torch.randn(1, 3, self.sample_size, self.sample_size)
        _, c, h, w = self.encode(dummy).shape
        self.latent_size = utils.LatentSize(c=c, h=h, w=w)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class MultiStageVGGAE(nn.ModuleList):
    def __init__(self):
        # store in inverse order for convenience (transport access is reversed)
        super().__init__([AutoEncoder(i) for i in range(5,0,-1)])
        self.preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        self.normalize = torch.nn.Identity()
        self.denormalize = torch.nn.Identity()
