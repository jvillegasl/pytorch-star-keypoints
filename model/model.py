from torch import Tensor
import torch.nn as nn
from base import BaseModel
from model.backbone import Backbone
from model.layers import MLP, PositionEmbeddingSine
from model.transformer import Transformer
from utils import nested_tensor_from_tensor_list


class DETRKpts(BaseModel):
    hidden_dim: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4

    def __init__(self, num_kpts):
        super().__init__()

        self.backbone = Backbone(name='resnet18')
        self.conv = nn.Conv2d(self.backbone.num_channels, self.hidden_dim, 1)
        self.position_embedding = PositionEmbeddingSine(
            self.hidden_dim//2,
            normalize=True
        )

        self.transformer = Transformer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=2048,
            normalize_before=True,
            return_intermediate_dec=True
        )

        self.visibility_embed = nn.Linear(self.hidden_dim, 1)
        self.kpts_embed = MLP(self.hidden_dim, self.hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_kpts, self.hidden_dim)

    def forward(self, input: tuple[Tensor, Tensor] | list[Tensor] | Tensor):
        if not isinstance(input, tuple):
            input = nested_tensor_from_tensor_list(input)

        features = self.backbone(input)

        src, mask = features
        assert mask is not None

        pos = self.position_embedding(features).to(src.dtype)

        src = self.conv(src)
        hs = self.transformer(src, mask, self.query_embed.weight, pos)[0]

        outputs_visibility = self.visibility_embed(hs).sigmoid()
        outputs_kpts = self.kpts_embed(hs).sigmoid()

        out = {
            'pred_kpts': outputs_kpts[-1],
            'pred_visibility': outputs_visibility[-1]
        }

        return out
