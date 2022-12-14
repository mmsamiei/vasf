from torch import nn
import einops
from utils.utils import positionalencoding2d, positionalencoding1d
from torch.nn import functional as F
import torch
#
class ImageGenerator(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers=1, nhead=8):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, hid_dim)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hid_dim, nhead=nhead, batch_first=True), num_layers=num_layers)
        #self.transformer_output_embedding = nn.Linear(hid_dim, hid_dim)
        self.modules = [
            nn.ConvTranspose2d(
                hid_dim, hid_dim, kernel_size=5, stride=1, padding=2, output_padding=0,
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                hid_dim, hid_dim, kernel_size=3, stride=1, padding=1, output_padding=0,
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hid_dim, 4, kernel_size=3, stride=1, padding=1, output_padding=0,)
        ]
        self.decoder = nn.Sequential(*self.modules)
    
    def forward(self, caption_rpr, image_size, token_mask=None):
        ### [b, l, d] -> [b, 3, 28, 28]
        batch_size, len_caption, dim = caption_rpr.shape
        h, w = image_size
        temp = self.input_embedding(caption_rpr)
        position_embd_1d = positionalencoding1d(dim, len_caption).to(next(self.parameters()).device)
        temp = temp + position_embd_1d
        temp = self.transformer(temp)
        temp = einops.repeat(caption_rpr, 'b l d -> (b l) d h w', h=h, w=w)
        position_embd_2d = positionalencoding2d(dim, h, w).to(next(self.parameters()).device)
        temp = temp + position_embd_2d
        temp = self.decoder(temp)
        temp = einops.rearrange(temp, '(b l) c w h -> b l c w h', b = batch_size)
        recons = temp[:, :, :3, :, :]
        masks = temp[:, :, -1:, :, :]
        masks[token_mask == 0] = -torch.inf
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        result = {
            'output': recon_combined,
            'token_outputs': recons,
            'masks': masks
        }
        return result