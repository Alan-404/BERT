import torch
import torch.nn as nn

from .utils.position import PositionalEncoding
from .modules.classification import Classification
from .modules.encoder import Encoder
from .utils.masking import generate_padding_mask

from typing import Optional

class BERT(nn.Module):
    def __init__(self,
                 n_tokens: int,
                 n_layers: int = 12,
                 d_model: int = 768,
                 n_heads: int = 12,
                 activation: str = 'relu',
                 dropout_p: float = 0.) -> None:
        super().__init__()

        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pe = PositionalEncoding(d_model)
        self.encoder = Encoder(n_layers, d_model, n_heads, activation, dropout_p)
        self.classification = Classification(d_model, n_tokens)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        batch_size, length = x.size()
        if lengths is not None:
            mask = generate_padding_mask(lengths).unsqueeze(1).unsqueeze(1)
        else:
            mask = torch.ones((batch_size,1, 1, length), dtype=torch.bool, device=x.device)

        x = self.embedding(x)
        x += self.pe(x.size(1))
        x = self.encoder(x, ~mask)
        x = self.classification(x)

        return x