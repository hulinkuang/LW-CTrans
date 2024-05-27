import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath

from lwctrans.network_architecture.Coformer.light_transformer_block import Mlp, LayerNorm


class SABlock(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if dim % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(dim, dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.input_rearrange = rearrange(x, "b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        # self.out_rearrange = rearrange(x, "b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):

        output = rearrange(self.qkv(x), "b h (qkv l d) -> qkv b l h d", qkv=3, l=self.num_heads)
        q, k, v = output[0], output[1], output[2]
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = rearrange(x, "b h l d -> b l (h d)")
        x = self.out_proj(x)
        x = self.drop_output(x)

        return x


class TransformerBlock(nn.Module):

    def __init__(self, dim,
                 num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim, data_format="channels_last")
        self.norm2 = norm_layer(dim, data_format="channels_last")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.token_mixer = SABlock(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout_rate=attn_drop)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b (d h w) c', b=B, c=C, d=D, h=H, w=W)
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = rearrange(x, 'b (d h w) c -> b c d h w', b=B, c=C, d=D, h=H, w=W)
        return x


if __name__ == '__main__':
    x = torch.randn(2, 32, 16, 40, 40)
    model = TransformerBlock(32)
    out = model(x)
    print(out.shape)
