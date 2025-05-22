import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

#learn f(x) such that x is the wt and f(x) is the CTCFKO - then i have to clean and label in a different way.
#or x is the CTCFKO and f(x) is the DKO
class LastLayerNN(nn.Module):
    def __init__(self):
        super(LastLayerNN, self).__init__()
        self.net = nn.Sequential(nn.Linear(83, 2),
            nn.GELU(),
            nn.Softmax(dim=-1),
            )
    def forward(self, x1, x2):
        return self.net(x1-x2)

class SiameseNet(nn.Module):
    def __init__(self, mask=False):
        super(SiameseNet, self).__init__()
        if mask:
            mask = np.tril(np.ones(256), k=-3)+np.triu(np.ones(256), k=3)
            self.mask = nn.Parameter(torch.tensor([mask], dtype=torch.int32), requires_grad = False)
    def mask_data(self, x):
        if hasattr(self, "mask"): x=torch.mul(self.mask, x)
        return x
    def forward_one(self, x):
        raise NotImplementedError
    def forward(self, x1, x2):
        x1, x2 = self.mask_data(x1), self.mask_data(x2)
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

class SLeNet(SiameseNet):
    def __init__(self, *args, **kwargs):
        super(SLeNet, self).__init__(*args, **kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5, 1),
            nn.MaxPool2d(2, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(16*61*61, 120),
            nn.GELU(),
            nn.Linear(120, 83),
            nn.GELU(),
            )
        self.distance = nn.CosineSimilarity()
    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

class SAlexNet(SiameseNet):
    def __init__(self, *args, **kwargs):
        super(SAlexNet, self).__init__(*args, **kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.GELU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.GELU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.GELU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.GELU(),
            nn.Linear(in_features=4096, out_features=83),
        )
        self.distance = nn.CosineSimilarity()
    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x

class SZFNet(SiameseNet):
    def __init__(self, *args, **kwargs):
        super(SZFNet, self).__init__(*args, **kwargs)
        self.channels = 1
        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()
    def get_conv_net(self):
        layers = []
        # in_channels = self.channels, out_channels = 96
        # kernel_size = 7x7, stride = 2
        layer = nn.Conv2d(
            self.channels, 96, kernel_size=7, stride=2, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))
        # in_channels = 96, out_channels = 256
        # kernel_size = 5x5, stride = 2
        layer = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))
        # in_channels = 256, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        # in_channels = 384, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        # in_channels = 384, out_channels = 256
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.GELU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        return nn.Sequential(*layers)
    def get_fc_net(self):
        layers = []
        # in_channels = 9216 -> output of self.conv_net
        # out_channels = 4096
        layer = nn.Linear(256*7*7, 4096)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Dropout())
        # in_channels = 4096
        # out_channels = self.class_count
        layer = nn.Linear(4096, 83)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Dropout())
        return nn.Sequential(*layers)
    def forward_one(self, x):
        y = self.conv_net(x)
        y = y.view(-1, 7*7*256)
        y = self.fc_net(y)
        return y

import torch
import torch.nn as nn
from torchvision.models import maxvit_t, MaxVit_T_Weights

class MaxVitEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        # 1) Load the pretrained MaxViT-Tiny
        base = maxvit_t(weights=MaxVit_T_Weights.IMAGENET1K_V1)
        
        # 2) Reuse its stem and transformer blocks
        self.stem   = base.stem            # initial Conv2dNormActivation layers
        self.blocks = base.blocks          # ModuleList of MaxVitBlock stages
        
        # 3) Reuse its classification head’s pooling+flatten+norm
        #    base.classifier is: [AdaptiveAvgPool2d, Flatten, LayerNorm, Linear, Tanh, Linear]
        self.pool    = base.classifier[0]  # AdaptiveAvgPool2d(1)
        self.flatten = base.classifier[1]  # Flatten()
        self.norm    = base.classifier[2]  # LayerNorm(512)
        
        # 4) We drop everything after the norm (i.e. the Linear+Tanh+Linear)
        #    The embedding dimension is simply the LayerNorm’s size:
        self.embedding_dim = self.norm.normalized_shape[0]  # == 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                # → (B, C0, H0, W0)
        for block in self.blocks:
            x = block(x)                # → (B, C_last, H_last, W_last)
        x = self.pool(x)                # → (B, C_last, 1, 1)
        x = self.flatten(x)             # → (B, C_last)
        x = self.norm(x)                # → (B, 512)
        return x                        # your embedding vector

# ————— Usage —————

# Move to GPU and eval mode
model = MaxVitEmbedder().cuda().eval()

# Dummy inference
x = torch.randn(8, 3, 224, 224, device='cuda')
with torch.no_grad():
    embeddings = model(x)
print("Embedding shape:", embeddings.shape)  # torch.Size([8, 512])

class MaxViTMultiScaleEmbedder(nn.Module):
    """
    A MaxViT‑Tiny backbone that returns a concatenated embedding
    of pooled features from each of its 4 transformer blocks.
    """
    def __init__(self, pretrained: bool = True, embed_dim: int | None = None):
        super().__init__()
        # 1) Load base model
        weights = MaxVit_T_Weights.IMAGENET1K_V1 if pretrained else None
        base = maxvit_t(weights=weights)

        # 2) Reuse stem + blocks
        self.stem   = base.stem            # initial conv layers
        self.blocks = base.blocks          # ModuleList of 4 MaxVitBlock

        # 3) One pool & flatten per block
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)       # (B, C, 1, 1) → (B, C)

        # 4) Explicit channel sizes for maxvit_t
        #    These are the default `block_channels` = [64, 128, 256, 512]
        channel_dims = [64, 128, 256, 512]
        concat_dim = sum(channel_dims)

        # 5) Optional projection head
        if embed_dim is not None:
            self.projector = nn.Linear(concat_dim, embed_dim, bias=False)
            self.output_dim = embed_dim
        else:
            self.projector = nn.Identity()
            self.output_dim = concat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) initial stem
        x = self.stem(x)

        # 2) iterate over blocks, pool each output
        feats = []
        for block in self.blocks:
            x = block(x)           # apply this transformer block
            f = self.pool(x)            # → (B, C_i, 1, 1)
            f = self.flatten(f)    # → (B, C_i)
            feats.append(f)

        # 3) concatenate all pooled features
        multiscale = torch.cat(feats, dim=1)  # (B, 960) for tiny

        # 4) optional projection
        emb = self.projector(multiscale)      # (B, output_dim)
        return emb

model = MaxViTMultiScaleEmbedder(pretrained=True, embed_dim=256).cuda().eval()
print("Output dim:", model.output_dim)   # 256

x = torch.randn(8, 3, 224, 224, device='cuda')
with torch.no_grad():
    out = model(x)
print("Multiscale embedding shape:", out.shape)  # torch.Size([8, 256])

