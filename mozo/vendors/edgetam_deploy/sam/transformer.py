# SPDX-License-Identifier: Apache-2.0
"""The two-way transformer the mask decoder runs its tokens through.

Taken from ``sam2/modeling/sam/transformer.py`` in ``facebookresearch/EdgeTAM``, which is SAM 2's
file with EdgeTAM's ``RoPEAttentionv2`` added. Three classes are carried -- the transformer, its
block, and the attention layer they share -- and the rest is left behind:

- ``RoPEAttention`` and ``RoPEAttentionv2`` are memory attention. Neither is reachable from an
  image, and the second is EdgeTAM's own addition for the 2-D spatial perceiver.
- ``sdp_kernel_context`` and ``get_sdpa_settings`` pin the attention backend, and the
  ``try``/``except``/``global`` fallback around them retries with the pin lifted. **This one does
  move numbers**, so it is worth being precise about what dropping it costs.

  ``get_sdpa_settings`` reads CUDA capability at import time and returns
  ``(OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON)``. On any machine without a modern CUDA GPU -- a
  Mac, a CPU box, a Jetson -- that is ``(True, False, True)``, which pins SDPA to the math
  kernel. On a recent CUDA GPU it selects flash instead. So upstream's own logits differ between
  a laptop and an A100, and "bit-exact against upstream" is a claim about a machine class rather
  than about a model.

  Dropping the pin costs 2e-07 per attention layer against upstream-on-CPU, compounding to 9e-05
  on the decoder's mask logits. Every other stage -- preprocessing, trunk, neck, prompt encoder,
  all of the decoder's own arithmetic -- is bit-identical, which is how that number was
  attributed: neutralise this context in the reference and all seven of the gate's prompts agree
  exactly. ``tools/verify/edgetam.py`` does exactly that, and says so.

  It is dropped rather than carried for four reasons: it makes the answer depend on what card is
  in the machine; ``torch.backends.cuda.sdp_kernel`` is deprecated in torch 2.11; it probes a
  device at module scope, so importing this package would touch hardware before mozo had chosen
  any; and the kernel it pins is the slower one -- 38.7 ms against 33.0 ms per decode here, a
  17 percent tax for a difference in the fifth decimal place.
  :mod:`~mozo.vendors.sam2_deploy` made the same call on the same code.
- The module-level ``warnings.simplefilter("ignore", FutureWarning)`` went with them. A vendored
  module should not silence a category of warning for the whole process that imports it.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ..layers import MLP

__all__ = ["Attention", "TwoWayAttentionBlock", "TwoWayTransformer"]


class Attention(nn.Module):
    """Multi-head attention, optionally computed at a narrower width than it takes in.

    Args:
        embedding_dim: Width of the queries, and of the output.
        num_heads: Heads to split the internal width across.
        downsample_rate: Divides *embedding_dim* to get the internal width. The cross-attention
            layers use 2; self-attention uses 1.
        dropout: Attention dropout. Zero in every published configuration, and inactive in
            ``eval`` regardless.
        kv_in_dim: Width of the keys and values, when it differs from *embedding_dim*.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        return self.out_proj(self._recombine_heads(out))


class TwoWayAttentionBlock(nn.Module):
    """One round of tokens and image attending to each other.

    Four layers, in order: self-attention over the tokens, tokens attending to the image, an MLP
    on the tokens, and the image attending back to the tokens.

    Args:
        embedding_dim: Channel width.
        num_heads: Heads in each attention layer.
        mlp_dim: Hidden width of the MLP.
        activation: Activation inside the MLP.
        attention_downsample_rate: Width divisor for the two cross-attention layers.
        skip_first_layer_pe: Omit the positional embedding on the self-attention, which the first
            block does because its queries *are* the positional embedding.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):
    """A stack of :class:`TwoWayAttentionBlock`, closed by one more token-to-image attention.

    Args:
        depth: How many blocks.
        embedding_dim: Channel width of the image embedding.
        num_heads: Heads in each attention layer.
        mlp_dim: Hidden width of each block's MLP.
        activation: Activation inside those MLPs.
        attention_downsample_rate: Width divisor for the cross-attention layers.
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self, image_embedding: Tensor, image_pe: Tensor, point_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Attend tokens and image to each other.

        Args:
            image_embedding: ``(B, C, H, W)``.
            image_pe: Positional encoding the shape of *image_embedding*.
            point_embedding: ``(B, N, C)`` prompt tokens.

        Returns:
            The processed tokens and the processed image, both flattened to ``(B, N, C)``.
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe
            )

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys
