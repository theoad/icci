import torch
import math
from einops import rearrange
from torchvision.models import ResNet, VisionTransformer, MobileNetV2, EfficientNet

def resnet34_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    return x  # [b, 256, 14, 14]

def vit_b_16_forward(self, x):
    x = self._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = self.encoder(x)
    x = x[:, 1:]  # removing cls token, with vitb16 [b, 196, 768]
    return rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.size(1)))).contiguous()  # [b, 768, 14, 14]

def mobilenet_v2_forward(self, x):
    return self.features[:-1](x)  # [b, 256, 12, 12]

def efficientnet_v2_s_forward(self, x):
    return self.features[:-1](x)  # [b, 256, 12, 12]

def remove_unused_params(model):
    if isinstance(model, ResNet):
        del model.layer4
        del model.avgpool
        del model.fc
    elif isinstance(model, VisionTransformer):
        del model.heads
    elif isinstance(model, (MobileNetV2, EfficientNet)):
        del model.features[-1]
        del model.classifier
    return model
