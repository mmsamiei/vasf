from torch import nn
import torch
import einops 

class AutoregressiveDescriptor(nn.Module):
    def __init__(self, hid_dim, input_dim, output_dim, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=256, nhead=8):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, hid_dim)
        self.transformer = nn.Transformer(d_model=hid_dim, nhead=8, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, batch_first=True)
        self.start_token = nn.parameter.Parameter(data=torch.randn(hid_dim), requires_grad=True)
        self.output_embedding = nn.Linear(hid_dim, output_dim)
    
    def forward(self, x, description_length):
        ## [batch, w, h, dim] -> [batch, l, dim]
        mask = torch.ones((x.shape[0], description_length))
        temp = x
        temp = einops.rearrange(temp, 'b w h d -> b (w h) d')
        temp = self.input_embedding(temp)
        src = temp
        start_token = einops.repeat(self.start_token, 'd -> b l d', b=src.shape[0], l=1)
        tgt = start_token
        for i in range(description_length):
            last_token = self.transformer(src, tgt)[:,-1:,:]
            print(last_token.shape)
            tgt = torch.cat([tgt, last_token], dim=1)
        tgt = tgt[:,1:,:]
        tgt = self.output_embedding(tgt)
        return tgt
    

class AutoregressiveMaskedDescriptor(nn.Module):
    def __init__(self, hid_dim, input_dim, output_dim, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=256, nhead=8):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, hid_dim)
        self.transformer = nn.Transformer(d_model=hid_dim, nhead=8, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, batch_first=True)
        self.start_token = nn.parameter.Parameter(data=torch.randn(hid_dim), requires_grad=True)
        self.output_embedding = nn.Linear(hid_dim, output_dim)

    def forward(self, x, description_length, threshold=1e-2):
        ## [batch, w, h, dim] -> [batch, l, dim]
        mask = torch.ones((x.shape[0], description_length))
        temp = x
        temp = einops.rearrange(temp, 'b w h d -> b (w h) d')
        temp = self.input_embedding(temp)
        src = temp
        start_token = einops.repeat(self.start_token, 'd -> b l d', b=src.shape[0], l=1)
        tgt = start_token
        for i in range(description_length):
            last_token = self.transformer(src, tgt)[:,-1:,:]
            last_token_is_valid = last_token.norm(dim=-1) > threshold
            mask[:, i:] = mask[:, i:] * last_token_is_valid
            tgt = torch.cat([tgt, last_token], dim=1)
        last_token = self.transformer(src, tgt)[:,-1:,:]
        commitment_loss = last_token.norm(dim=-1)
        tgt = tgt[:,1:,:]
        tgt = self.output_embedding(tgt)
        return tgt, mask, commitment_loss