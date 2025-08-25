import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import maxvit_t, resnet50

#learn f(x) such that x is the wt and f(x) is the CTCFKO - then i have to clean and label in a different way.
#or x is the CTCFKO and f(x) is the DKO
class LastLayerNN(nn.Module):
    def __init__(self, triplet=False):
        super(LastLayerNN, self).__init__()
        self.net = nn.Sequential(nn.LazyLinear(2),
            nn.GELU(),
            nn.Softmax(dim=-1),
            )
    def forward(self, x1, x2):
        return self.net(x1-x2)

class SiameseNet(nn.Module):
    def __init__(self, mask=False, image_size=224, triplet=False, supcon=False):
        super(SiameseNet, self).__init__()
        self.triplet = triplet
        self.supcon = supcon
        if mask:
            mask = np.tril(np.ones(image_size), k=-3) + np.triu(np.ones(image_size), k=3)
            self.mask = nn.Parameter(torch.tensor(mask, dtype=torch.int32), requires_grad=False)

    def mask_data(self, x):
        if hasattr(self, "mask"):
            x = torch.mul(self.mask, x)
        return x

    def normalize_data(self, x):
        """Normalize batch of tensors (B,C,H,W) to [0, 1] by dividing by max."""
        if torch.is_tensor(x):
            max_vals = x.amax(dim=(1, 2, 3), keepdim=True)
            return x / max_vals  # avoid division by zero
        else:
            raise TypeError("Input must be a PyTorch tensor")

    def forward_one(self, x):
        raise NotImplementedError

    def forward(self, x1, x2, x3=None):
        # Apply normalization if in eval/test mode

        x1 = self.normalize_data(x1)
        x2 = self.normalize_data(x2)
        if x3 is not None:
            x3 = self.normalize_data(x3)

        if self.triplet:  # for triplet loss
            x1, x2, x3 = self.mask_data(x1), self.mask_data(x2), self.mask_data(x3)
            return self.forward_one(x1), self.forward_one(x2), self.forward_one(x3)
        elif self.supcon:
            pass
        else:  # for regular contrastive loss
            x1, x2 = self.mask_data(x1), self.mask_data(x2)
            return self.forward_one(x1), self.forward_one(x2)

class SLeNet(SiameseNet):
    def __init__(self, *args, mask=False, image_size=224, triplet=False, supcon=False):
        super().__init__(mask=mask, image_size=image_size, triplet=triplet, supcon=supcon)

        # same feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1,6,5,1),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5,1),
            nn.MaxPool2d(2,2),
        )

        # infer flattened dim
        with torch.no_grad():
            dummy = torch.zeros((1,)+(1,image_size,image_size))
            feat = self.features(dummy)
            feat_dim = int(feat.view(1, -1).size(1))

        # projection head
        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 120),
            nn.GELU(),
            nn.Linear(120, 83),
            nn.GELU(),
        )
        
    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        # only for SupCon / SINCERELoss and triplet
        if self.supcon:
            x = F.normalize(x, p=2, dim=1)

        return x

# For testing the original models

class SiameseNet_ediem_test(nn.Module):
    def __init__(self, mask=False, image_size=256, triplet=False, supcon=False):
        super(SiameseNet_ediem_test, self).__init__()
        self.triplet = triplet
        self.supcon = supcon
        if mask:
            # shape: (1, H, W) to match checkpoint
            mask_array = np.tril(np.ones(image_size), k=-3) + np.triu(np.ones(image_size), k=3)
            mask_tensor = torch.tensor(mask_array, dtype=torch.int32).unsqueeze(0)  # add channel dim
            self.mask = nn.Parameter(mask_tensor, requires_grad=False)

    def mask_data(self, x):
        if hasattr(self, "mask"):
            x = torch.mul(self.mask, x)
        return x

    def normalize_data(self, x):
        """Normalize batch of tensors (B,C,H,W) to [0, 1] by dividing by max."""
        if torch.is_tensor(x):
            max_vals = x.amax(dim=(1, 2, 3), keepdim=True)
            return x / max_vals  # avoid division by zero
        else:
            raise TypeError("Input must be a PyTorch tensor")

    def forward_one(self, x):
        raise NotImplementedError

    def forward(self, x1, x2, x3=None):
        # Apply normalization if in eval/test mode

        x1 = self.normalize_data(x1)
        x2 = self.normalize_data(x2)
        if x3 is not None:
            x3 = self.normalize_data(x3)

        if self.triplet:  # for triplet loss
            x1, x2, x3 = self.mask_data(x1), self.mask_data(x2), self.mask_data(x3)
            return self.forward_one(x1), self.forward_one(x2), self.forward_one(x3)
        elif self.supcon:
            pass
        else:  # for regular contrastive loss
            x1, x2 = self.mask_data(x1), self.mask_data(x2)
            return self.forward_one(x1), self.forward_one(x2)


class SLeNet_ediem_test(SiameseNet_ediem_test):
    def __init__(self, *args, mask=False, image_size=224, triplet=False, supcon=False):
        super().__init__(mask=mask, image_size=image_size, triplet=triplet, supcon=supcon)

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1),
            nn.MaxPool2d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, 1, image_size, image_size))
            feat = self.features(dummy)
            feat_dim = int(feat.view(1, -1).size(1))

        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 120),
            nn.GELU(),
            nn.Linear(120, 83),
            nn.GELU(),
        )

        self.distance = nn.CosineSimilarity()

    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        if self.supcon:
            x = F.normalize(x, p=2, dim=1)
        return x

# Didin't train well
class SAlexNet(SiameseNet):
    def __init__(self, *args, mask=False, image_size=224, triplet=False, supcon=False):
        super().__init__(mask=mask, image_size=image_size, triplet=triplet, supcon=supcon)
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.GELU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.GELU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Infer flattened feature dim using dummy input
        with torch.no_grad():
            dummy = torch.zeros((1, 1, image_size, image_size))
            feat = self.features(dummy)
            feat_dim = int(feat.view(1, -1).size(1))

        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 4096),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.GELU(),
            nn.Linear(4096, 512),
            nn.GELU()
        )

        self.norm = nn.LayerNorm(512)
        self.distance = nn.CosineSimilarity()

    def forward_one(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.norm(x)
        return x

# Didn't train well
class SResNet50(SiameseNet):
    def __init__(self, *args, mask=False, image_size=224, triplet=False, supcon=False):
        super().__init__(mask=mask, image_size=image_size, triplet=triplet, supcon=supcon)
        # load ResNet50 backbone without final FC
        backbone = resnet50(pretrained=False)
        # use all layers except the final fully-connected
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # includes avgpool

        # infer flattened feature dim using dummy input
        with torch.no_grad():
            dummy = torch.zeros((1, 3, image_size, image_size))
            feat = self.features(dummy)
            feat_dim = int(feat.view(1, -1).size(1))

        # build a flexible linear head
        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 4096),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.GELU()
        )

        self.norm = nn.LayerNorm(512)
        self.distance = nn.CosineSimilarity()

    def forward_one(self, x):
        # if input is single-channel, expand to 3
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.norm(x)
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

class MaxVitEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        base = maxvit_t(weights=None)
        self.stem = base.stem
        self.blocks = base.blocks
        self.pool = base.classifier[0]  # AdaptiveAvgPool2d
        self.flatten = base.classifier[1]  # Flatten
        self.embedding_dim = base.classifier[2].normalized_shape[0]  # 512

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x  # (B, 512)


class SMaxVit(SiameseNet):
    """Siamese network using MaxViT transformer backbone without projection head."""
    def __init__(self, mask=False, image_size=224, triplet=False, supcon=False):
        super().__init__(mask=mask, image_size=image_size, triplet=triplet, supcon=supcon)
        self.embedder = MaxVitEmbedder()

    def forward_one(self, x):
        # Repeat single-channel to 3 if needed (make into color essentially)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.embedder(x)  # (B, 512)

        # Only L2-normalize for SupCon / SINCERELoss
        if self.supcon:
            x = F.normalize(x, p=2, dim=1)

        return x


class MaxViTMultiScaleEmbedder(nn.Module):
    """
    MaxViT-Tiny backbone that returns pooled multiscale features.
    Pools from each of the 4 transformer blocks, concatenates, and
    projects to 512 dimensions.
    """
    def __init__(self, output_dim: int = 512):
        super().__init__()
        base = maxvit_t(weights=None)

        # Stem + transformer blocks
        self.stem   = base.stem
        self.blocks = base.blocks

        # Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)  # (B, C, 1, 1) → (B, C)

        # Channel sizes for maxvit_t
        channel_dims = [64, 128, 256, 512]
        concat_dim = sum(channel_dims)  # 960 for tiny

        # Projection to fixed dim (512 by default)
        self.projector = nn.Linear(concat_dim, output_dim, bias=False)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        feats = []
        for block in self.blocks:
            x = block(x)
            f = self.pool(x)
            f = self.flatten(f)
            feats.append(f)

        multiscale = torch.cat(feats, dim=1)      # (B, 960)
        emb = self.projector(multiscale)          # (B, 512)
        return emb

class SMaxViTMulti(SiameseNet):
    """Siamese wrapper for the 512-dim MaxViT multiscale embedder."""
    def __init__(self, mask=False, image_size=224, triplet=False, supcon=False):
        super().__init__(mask=mask, image_size=image_size, triplet=triplet, supcon=supcon)
        self.embedder = MaxViTMultiScaleEmbedder(output_dim=512)

    def forward_one(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.embedder(x)  # (B, 512)
        if self.supcon:
            x = F.normalize(x, p=2, dim=1)
        return x
