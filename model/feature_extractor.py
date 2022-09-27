import torch.nn as nn
import einops
from utils import utils

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class CNNFeatureExtractor(nn.Module):
    def __init__(self, hid_dim=64):
        super(CNNFeatureExtractor, self).__init__()
        self.conv0= nn.Conv2d(3, hid_dim, 1)
        self.encoder = nn.Sequential(
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
        )

    def forward(self, x):
        temp = self.conv0(x)
        temp = self.encoder(temp) # [b d h w]
        _, c, h, w = temp.shape
        temp = temp + utils.positionalencoding2d(c, h, w).to(next(self.parameters()).device)
        temp = einops.rearrange(temp, 'b c h w -> b h w c')
        return temp