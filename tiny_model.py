"""
TinyModel — Recurrent-Depth Transformer with Linear Attention
============================================================

Architecture:
    Input → [Prelude P] → [Recurrent Block R] (looped T times) → [Coda C] → Output

Key features:
- Kim Linear Attention (no softmax quadratic complexity)
- Loop-index embedding so same weights handle different loop iterations
- ACT halting: complex tokens loop more, simple tokens loop less
- No MoE (dense FFN only)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyConfig:
    vocab_size: int = 8192
    dim: int = 512
    n_heads: int = 8
    max_seq_len: int = 1024
    max_loop_iters: int = 8
    prelude_layers: int = 2
    coda_layers: int = 2
    act_threshold: float = 0.9
    rope_theta: float = 10000.0
    lora_rank: int = 24
    dropout: float = 0.1
    tie_weights: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return (
        torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2))
        .flatten(-2)
        .to(x.dtype)
    )


class LinearAttention(nn.Module):
    """
    Kim-style Linear Attention — gating-based, no softmax quadratic cost.

    Inspired by RetNet / RWKV / Linear Transformers.
    Uses gating mechanism instead of softmax attention:
        y = (W_v * x) ⊙ sigmoid(W_g * x)  accumulated via linear recurrence
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        H = self.n_heads
        d = self.head_dim

        q = self.q_proj(x).view(B, T, H, d)
        k = self.k_proj(x).view(B, T, H, d)
        v = self.v_proj(x).view(B, T, H, d)
        g = torch.sigmoid(self.g_proj(x)).view(B, T, H, d)

        k = F.elu(k) + 1
        v = v * g

        if state is None:
            state = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)

        new_state = torch.zeros(B, H, d, d, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            k_t = k[:, t]  # (B, H, d)
            v_t = v[:, t]  # (B, H, d)

            state = state + torch.einsum("bhd,bhd->bhd", k_t, v_t).unsqueeze(-1)
            y_t = torch.einsum("bhd,hde->bhe", state.squeeze(-1), q[:, t]) / (d ** 0.5)
            outputs.append(y_t.unsqueeze(1))

        out = torch.cat(outputs, dim=1).reshape(B, T, H * d)
        return self.o_proj(out), state


class GatedLinearAttention(nn.Module):
    """
    Gated Linear Attention with retention and learnable state decay.
    s_t = λ * s_{t-1} + k_t ⊗ v_t
    y_t = s_t @ q_t / sqrt(d)
    
    The decay λ ensures the state doesn't grow unbounded, stabilizing
    gradients across long sequences.
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.log_decay = nn.Parameter(torch.zeros(n_heads))
        self._decay_min = math.log(0.01)
        self._decay_max = math.log(0.999)

    def _get_decay(self, T: int, device: torch.device) -> torch.Tensor:
        decay = torch.sigmoid(self.log_decay)  # (H,) in (0,1)
        decay = 0.99 + 0.009 * decay  # range [0.99, 0.999)
        return decay

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        d = self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, d)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        g = torch.sigmoid(self.gate(x)).view(B, T, H, d)

        k = F.elu(k) + 1
        v = v * g

        if freqs_cis is not None:
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)

        decay = self._get_decay(T, x.device)  # (H,)

        ts = torch.arange(T, device=x.device, dtype=x.dtype)  # (T,)
        decay_pow = decay.unsqueeze(0) ** ts.unsqueeze(-1)  # (T, H)
        inv_decay_pow = (1.0 / decay).unsqueeze(0) ** ts.unsqueeze(-1)  # (T, H)

        k_scaled = k * inv_decay_pow.unsqueeze(-1)  # (B, T, H, d)
        kv = k_scaled.unsqueeze(-1) * v.unsqueeze(-2)  # (B, T, H, d, d)
        kv_cumsum = kv.cumsum(dim=1)
        s = kv_cumsum * decay_pow.unsqueeze(-1).unsqueeze(-1)  # (B, T, H, d, d)

        y = torch.einsum("bthde,bthe->bthd", s, q) / (d ** 0.5)
        out = y.reshape(B, T, H * d)
        return self.o_proj(out)


class StandardAttention(nn.Module):
    """Standard GQA attention as fallback."""

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)

        if freqs_cis is not None:
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    """Dense SwiGLU FFN (no MoE)."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4 // 3
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def loop_index_embedding(
    h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """Inject sinusoidal loop-index signal into hidden state."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


class LoRAAdapter(nn.Module):
    """Depth-wise LoRA for per-loop adaptation."""

    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.scale = nn.Embedding(max_loops, rank)

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        max_t = self.scale.num_embeddings - 1
        t_idx = loop_t if loop_t <= max_t else max_t
        s = self.scale(torch.tensor(t_idx, device=x.device))
        down = self.down(x) * s
        return down @ self.B


class LTIInjection(nn.Module):
    """Stable input injection with spectral radius < 1."""

    def __init__(self, dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor
    ) -> torch.Tensor:
        A = self.get_A()
        return A * h + self.B * e + transformer_out


class ACTHalting(nn.Module):
    """Adaptive Computation Time halting."""

    def __init__(self, dim: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.halt(h)).squeeze(-1)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block with linear attention option."""

    def __init__(self, cfg: TinyConfig, use_linear_attn: bool = True):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)

        if use_linear_attn and cfg.dim >= 256:
            self.attn = GatedLinearAttention(cfg.dim, cfg.n_heads)
        else:
            self.attn = StandardAttention(cfg)

        self.ffn = FeedForward(cfg.dim)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.resid_drop(self.attn(self.attn_norm(x), freqs_cis))
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


class RecurrentBlock(nn.Module):
    """Looped transformer block with ACT and input injection."""

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_linear_attn=True)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = cfg.dim // 4

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
    ) -> torch.Tensor:
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            trans_out = self.block(combined, freqs_cis, mask)
            trans_out = trans_out + self.lora(trans_out, t)
            h = self.injection(h, e, trans_out)

            p = self.act(h)
            still_running = ~halted

            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            if halted.all() and mask is None:
                break

        return h_out


class TinyModel(nn.Module):
    """
    Recurrent-Depth Transformer with Linear Attention.

    Input → [Prelude] → [Recurrent Block] (looped T times) → [Coda] → Output
    """

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis", freqs)

        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_linear_attn=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_linear_attn=False) for _ in range(cfg.coda_layers)]
        )

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        T = input_ids.shape[1]
        device = input_ids.device

        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        mask = self._causal_mask(T, device, x.dtype) if T > 1 else None

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask)

        e = x
        x = self.recurrent(x, e, freqs_cis, mask, n_loops)

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask)

        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = input_ids.shape[1] - 1

            logits = self.forward(cur_ids, n_loops=n_loops, start_pos=start_pos)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

        return input_ids

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    cfg = TinyConfig()
    model = TinyModel(cfg)
    total = model.num_parameters()
    print(f"TinyModel parameters: {total:,}")
    print(f"Config: dim={cfg.dim}, heads={cfg.n_heads}, loops={cfg.max_loop_iters}")

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(ids, n_loops=4)
    print(f"Logits shape: {logits.shape}")

    out = model.generate(ids, max_new_tokens=8, n_loops=8)
    print(f"Generated shape: {out.shape}")

    A = model.recurrent.injection.get_A()
    print(f"Spectral radius ρ(A) max: {A.max().item():.4f}")