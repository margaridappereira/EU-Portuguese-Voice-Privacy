import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)


    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)















class TransformerDecoder(nn.Module):
    def __init__(
            self,
            encoder_embed_dim: int = 1024,
            hidden_dim: int = 256,
            num_heads: int = 4,
            num_layers: int = 4,
            dropout: float = 0.2, #0.1,
            output_dim: int = 46, 
            emb_dim: int = -1,
    ):
        super().__init__()
        self.emb_dim = emb_dim

        if emb_dim > 0:
            self.emb = nn.Embedding(emb_dim, encoder_embed_dim, padding_idx=47)

        self.input_proj = nn.Linear(encoder_embed_dim, hidden_dim) if encoder_embed_dim != hidden_dim else nn.Identity()
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, mask=None):
        if self.emb_dim > 0:
            x = self.emb(x)
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.proj(x)
        return x
    
def load_transformer_model(ckpt_path: str, device: str) -> TransformerDecoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    output_dim = ckpt["proj.weight"].shape[0]
    hidden_dim = ckpt["proj.weight"].shape[1]
    
    model = TransformerDecoder(
            encoder_embed_dim = 1024,
            output_dim = 46,
        ).to(device)
    
    import pdb; pdb.set_trace()
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    return model
