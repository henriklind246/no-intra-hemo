"""Time-conditioned LinearNO: linear-attention neural operator over the point cloud.

Ported from the chunk-capable LinearNO of the AhmedML benchmark
(models/LinearNO_chunk_opt_matrix_mul.py, itself following the official
HiPRL/LinearNO implementation). The attention is

    Q = softmax(Q / t_q) over features,   K = softmax(K / t_k) over points,
    out = Q (K^T V),

so every point attends to a softmax-weighted average over all points of its
field, in time linear in the number of points. Because the softmax over
points only enters through K^T V, it can be accumulated chunk by chunk
(``forward_field``): a field of any size is evaluated exactly, with every
point attending to the whole field, in bounded memory. In training the field
is a random subset of its points (``SampledPoints``); K^T V is then an
estimate of the same weighted average.

What is added for AneuG-Flow:

* Early geometry memory injection with FiLM (Xia and Aviles-Rivero, "Do
  Neural Operators Forget Geometry?", ICML 2026). A geometry encoder, a
  pointwise MLP on the original point inputs (coordinates and distance
  features), gives per-point gamma and beta. After each of the first
  ``injection_blocks`` blocks (default: the first half, blocks 0-3 of 8) the
  hidden state is modulated as ``x = gamma * x + beta``. The encoder and its
  gamma, beta are shared by all injection points, and do not depend on time
  (the geometry is static). Injecting only early follows the paper's finding
  that late injection lets the network bypass the attention (the "geometric
  shortcut"). The encoder's last layer starts at zero (gamma = 1, beta = 0).
* Time and flow-split conditioning. sin/cos harmonics of the phase and the
  flow split go through an MLP to a condition vector c, one per (case, time).
  Every block's two LayerNorms are modulated by c (scale and shift) and its
  two residual branches gated by c (adaptive LayerNorm / FiLM), and so is the
  final LayerNorm before the output layer. To keep the model small, one
  linear layer shared by all blocks turns c into the 6 shift/scale/gate
  vectors and each block adds its own learned table of 6 vectors
  ("adaLN-single", as in PixArt-alpha): 0.6M parameters for 8 x 256,
  condition MLP included, instead of 3.3M for a modulation layer per block.
  The modulation starts at zero, so at initialization the network is exactly
  the original LinearNO, and the conditioning is learned from there.

Both start as identities, so at initialization the network is exactly the
original LinearNO.

Default size: the efficient LinearNO of the AhmedML benchmark (8 blocks,
width 256, 8 heads, MLP ratio 2, key ratio 1; 3.85M parameters there). With
the conditioning (0.6M) and the geometry encoder (0.13M) it has 4.6M.
``geometry_injection=False`` gives the plain time-conditioned LinearNO.

Interface (as for every model in src.models):
    forward(inputs (B, K, P), flow_split (B, S), time (B, s)) -> (B, s, K, out)
    forward_field(list of (B, K_i, P) chunks, flow_split, time) -> list of (B, s, K_i, out)
With ``checkpoint_blocks`` each block is recomputed in the backward pass
instead of storing its activations (much less memory, about 30% more time).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .mlp import time_features

ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


def _mlp(n_in: int, n_hidden: int, n_out: int, act: str) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, n_hidden), ACTIVATIONS[act](), nn.Linear(n_hidden, n_out))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class LinearNOAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32, dropout: float = 0.0, key_ratio: int = 1):
        super().__init__()
        inner = heads * dim_head
        self.heads, self.dim_head = heads, dim_head
        self.key_dim = key_ratio * dim_head
        # Initial values as in the official LinearNO.
        self.temperature_q = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.temperature_k = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.in_project_x = nn.Linear(dim, inner)
        self.to_q = nn.Linear(dim_head, self.key_dim, bias=False)
        self.to_k = nn.Linear(dim_head, self.key_dim, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.GELU(), nn.Linear(dim, dim), nn.Dropout(dropout))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        """(R, N, C) -> (R, H, N, D)"""
        r, n, _ = x.shape
        return self.in_project_x(x).view(r, n, self.heads, self.dim_head).transpose(1, 2)

    def _k_logits(self, xm: torch.Tensor) -> torch.Tensor:
        return self.to_k(xm) / torch.clamp(self.temperature_k, 0.1, 2.0)

    def _attend(self, xm: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = F.softmax(self.to_q(xm) / torch.clamp(self.temperature_q, 0.1, 2.0), dim=-1)
        out = torch.einsum("rhnk,rhkd->rhnd", q, kv)
        r, h, n, d = out.shape
        return self.to_out(out.transpose(1, 2).reshape(r, n, h * d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xm = self._heads(x)
        k = F.softmax(self._k_logits(xm), dim=-2)  # over points
        kv = torch.einsum("rhnk,rhnd->rhkd", k, self.to_v(xm))
        return self._attend(xm, kv)

    # --- chunked: the softmax over points accumulated across chunks ---------------

    def chunk_kv_stats(self, x: torch.Tensor):
        """Local max, denominator and numerator of the softmax-weighted K^T V of one chunk."""
        xm = self._heads(x)
        logits = self._k_logits(xm)
        local_max = logits.amax(dim=2)
        e = torch.exp(logits - local_max.unsqueeze(2))
        return local_max, e.sum(dim=2), torch.einsum("rhnk,rhnd->rhkd", e, self.to_v(xm))

    @staticmethod
    def merge_kv_stats(g_max, g_den, g_num, l_max, l_den, l_num):
        if g_max is None:
            return l_max, l_den, l_num
        new_max = torch.maximum(g_max, l_max)
        a, b = torch.exp(g_max - new_max), torch.exp(l_max - new_max)
        return new_max, g_den * a + l_den * b, g_num * a.unsqueeze(-1) + l_num * b.unsqueeze(-1)

    def apply_global_kv(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        return self._attend(self._heads(x), kv)


class ConditionedLinearNOBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, *, mlp_ratio: int = 2, key_ratio: int = 1, act: str = "gelu",
                 dropout: float = 0.0, last_layer: bool = False, out_dim: int = 4):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden)
        self.attn = LinearNOAttention(hidden, heads=heads, dim_head=hidden // heads, dropout=dropout,
                                      key_ratio=key_ratio)
        self.ln_2 = nn.LayerNorm(hidden)
        self.mlp = _mlp(hidden, hidden * mlp_ratio, hidden, act)
        # This block's offsets to the shared shift/scale/gate (attention, MLP).
        self.modulation_table = nn.Parameter(torch.zeros(6, hidden))
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden)
            self.out = nn.Linear(hidden, out_dim)

    def zero_init_conditioning(self) -> None:
        nn.init.zeros_(self.modulation_table)

    def _mods(self, shared: torch.Tensor):
        """(R, 6, H) shared modulation -> 6 tensors of (R, 1, H) for this block."""
        return [m.unsqueeze(1) for m in (shared + self.modulation_table).unbind(dim=1)]

    def _after_attention(self, fx, attn_out, mods, final):
        _, _, g1, b2, s2, g2 = mods
        fx = fx + (1 + g1) * attn_out
        fx = fx + (1 + g2) * self.mlp(_modulate(self.ln_2(fx), b2, s2))
        if self.last_layer:
            b3, s3 = final[:, 0:1], final[:, 1:2]
            fx = self.out(_modulate(self.ln_3(fx), b3, s3))
        return fx

    def forward(self, fx: torch.Tensor, shared: torch.Tensor, final: torch.Tensor) -> torch.Tensor:
        """fx (R, N, H); shared (R, 6, H) block modulation; final (R, 2, H), used by the last block."""
        mods = self._mods(shared)
        attn_out = self.attn(_modulate(self.ln_1(fx), mods[0], mods[1]))
        return self._after_attention(fx, attn_out, mods, final)

    def forward_chunks(self, fx_list: List[torch.Tensor], shared: torch.Tensor, final: torch.Tensor,
                       eps: float = 1e-30) -> List[torch.Tensor]:
        """Exact chunked block: pass 1 accumulates the global K^T V, pass 2 applies it to every chunk."""
        mods = self._mods(shared)
        stats = (None, None, None)
        for f in fx_list:
            stats = self.attn.merge_kv_stats(*stats, *self.attn.chunk_kv_stats(_modulate(self.ln_1(f), mods[0],
                                                                                         mods[1])))
        kv = stats[2] / (stats[1].unsqueeze(-1) + eps)
        out = []
        for f in fx_list:
            attn_out = self.attn.apply_global_kv(_modulate(self.ln_1(f), mods[0], mods[1]), kv)
            out.append(self._after_attention(f, attn_out, mods, final))
        return out


class TimeConditionedLinearNO(nn.Module):
    def __init__(self, num_point_inputs: int = 8, num_flow_split: int = 2, num_outputs: int = 4, *,
                 n_hidden: int = 256, n_layers: int = 8, n_head: int = 8, mlp_ratio: int = 2, key_ratio: int = 1,
                 act: str = "gelu", dropout: float = 0.0, time_harmonics: int = 4, geometry_injection: bool = True,
                 injection_blocks: Optional[int] = None, checkpoint_blocks: bool = True):
        super().__init__()
        if act not in ACTIVATIONS:
            raise ValueError(f"act must be one of {sorted(ACTIVATIONS)}")
        if n_hidden % n_head:
            raise ValueError("n_hidden must be divisible by n_head")
        if injection_blocks is None:
            injection_blocks = n_layers // 2
        if geometry_injection and not 0 <= injection_blocks < n_layers:
            # The last block ends in the output layer, so there is no hidden state after it.
            raise ValueError(f"injection_blocks must be in [0, {n_layers - 1}] for {n_layers} blocks")
        self.config = dict(num_point_inputs=num_point_inputs, num_flow_split=num_flow_split,
                           num_outputs=num_outputs, n_hidden=n_hidden, n_layers=n_layers, n_head=n_head,
                           mlp_ratio=mlp_ratio, key_ratio=key_ratio, act=act, dropout=dropout,
                           time_harmonics=time_harmonics, geometry_injection=geometry_injection,
                           injection_blocks=injection_blocks, checkpoint_blocks=checkpoint_blocks)
        self.time_harmonics = time_harmonics
        self.checkpoint_blocks = checkpoint_blocks
        self.num_outputs = num_outputs
        self.preprocess = _mlp(num_point_inputs, 2 * n_hidden, n_hidden, act)  # lifting, as in LinearNO
        # Early geometry memory injection: original point inputs -> FiLM (delta gamma, beta), shared.
        self.geometry_encoder = _mlp(num_point_inputs, n_hidden, 2 * n_hidden, act) if geometry_injection else None
        self.inject_after = list(range(injection_blocks)) if geometry_injection else []
        self.placeholder = nn.Parameter((1.0 / n_hidden) * torch.rand(n_hidden))
        self.condition = nn.Sequential(nn.Linear(num_flow_split + 2 * time_harmonics, n_hidden),
                                       ACTIVATIONS[act](), nn.Linear(n_hidden, n_hidden), nn.SiLU())
        # Shared by all blocks (adaLN-single): condition -> 6 block and 2 final modulation vectors.
        self.modulation = nn.Linear(n_hidden, 6 * n_hidden)
        self.final_modulation = nn.Linear(n_hidden, 2 * n_hidden)
        self.n_hidden = n_hidden
        self.blocks = nn.ModuleList([
            ConditionedLinearNOBlock(n_hidden, n_head, mlp_ratio=mlp_ratio, key_ratio=key_ratio, act=act,
                                     dropout=dropout, last_layer=(i == n_layers - 1), out_dim=num_outputs)
            for i in range(n_layers)
        ])
        self.apply(self._init_weights)
        zero = [self.modulation, self.final_modulation]
        if self.geometry_encoder is not None:
            zero.append(self.geometry_encoder[-1])  # gamma = 1, beta = 0
        for layer in zero:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for b in self.blocks:
            b.zero_init_conditioning()

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _cond(self, flow_split: torch.Tensor, time: torch.Tensor):
        """(B, S), (B, s) -> shared block modulation (B * s, 6, H) and final modulation (B * s, 2, H)."""
        s = time.shape[1]
        g = torch.cat([flow_split[:, None, :].expand(-1, s, -1), time_features(time, self.time_harmonics)], -1)
        c = self.condition(g).reshape(-1, self.n_hidden)
        return (self.modulation(c).view(-1, 6, self.n_hidden),
                self.final_modulation(c).view(-1, 2, self.n_hidden))

    @staticmethod
    def _per_time(x: torch.Tensor, s: int) -> torch.Tensor:
        """(B, K, H) -> (B * s, K, H), the same geometry for each of the s times."""
        b, k, h = x.shape
        return x[:, None].expand(b, s, k, h).reshape(b * s, k, h)

    def _film(self, inputs: torch.Tensor, s: int):
        """(B, K, P) original point inputs -> gamma, beta (B * s, K, H), or None without injection."""
        if self.geometry_encoder is None:
            return None
        d_gamma, beta = self.geometry_encoder(inputs).chunk(2, dim=-1)
        return self._per_time(1 + d_gamma, s), self._per_time(beta, s)

    def _apply_film(self, fx: torch.Tensor, inputs: torch.Tensor, s: int) -> torch.Tensor:
        gamma, beta = self._film(inputs, s)
        return gamma * fx + beta

    def forward(self, inputs: torch.Tensor, flow_split: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        b, k, _ = inputs.shape
        s = time.shape[1]
        shared, final = self._cond(flow_split, time)
        film = self._film(inputs, s)
        fx = self._per_time(self.preprocess(inputs), s) + self.placeholder
        use_ckpt = self.checkpoint_blocks and self.training and torch.is_grad_enabled()
        for i, block in enumerate(self.blocks):
            fx = checkpoint(block, fx, shared, final, use_reentrant=False) if use_ckpt else block(fx, shared, final)
            if i in self.inject_after:
                fx = film[0] * fx + film[1]
        return fx.reshape(b, s, k, self.num_outputs)

    def forward_field(self, chunks: List[torch.Tensor], flow_split: torch.Tensor,
                      time: torch.Tensor) -> List[torch.Tensor]:
        """The whole field given as chunks of points: every point attends to all chunks (exact)."""
        b, s = flow_split.shape[0], time.shape[1]
        shared, final = self._cond(flow_split, time)
        fx = [self._per_time(self.preprocess(x), s) + self.placeholder for x in chunks]
        for i, block in enumerate(self.blocks):
            fx = block.forward_chunks(fx, shared, final)
            if i in self.inject_after:
                # FiLM is pointwise: recompute gamma, beta one chunk at a time instead of keeping the field's.
                fx = [self._apply_film(f, x, s) for f, x in zip(fx, chunks)]
        return [f.reshape(b, s, x.shape[1], self.num_outputs) for f, x in zip(fx, chunks)]
