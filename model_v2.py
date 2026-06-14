"""
model_v2.py — ConvergenceQwen

V2 (measurement, not structural sharing).
Unlike model.py, this keeps every layer's FULL weight matrix as a real
trainable parameter. We do NOT impose a shared backbone. Instead we add a soft
consolidation penalty (in distill_v2.py) that pulls each class's 24 matrices
toward their running mean, and we MEASURE how far they actually converge via
per-class spread. Per-layer rank-r adapters give the weights a cheap extra
degree of freedom to stay functional while the bulk consolidates.

    W_eff[layer i, class c] = W[i,c] + scale * (A[i,c] @ B[i,c])

  - W[i,c] : (out,in) FULL per-layer weight, init = teacher's actual weight
  - A,B    : per-layer rank-r adapter, A small-random / B zero (delta=0 at init)

At init W_eff == teacher exactly (not the mean) — we want to watch the weights
TRAVEL toward the mean under the penalty, so they must start where the teacher
put them.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

WEIGHT_CLASSES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
_ATTN_CLASSES = {"q_proj", "k_proj", "v_proj", "o_proj"}


class AdaptedLinear(nn.Module):
    """Full per-layer weight + a per-layer rank-r adapter. The weight is OWNED
    here (unlike V1's shared backbone) and is fully trainable."""
    def __init__(self, weight: torch.Tensor, rank: int, alpha: float,
                 bias: torch.Tensor | None):
        super().__init__()
        out_features, in_features = weight.shape
        self.weight = nn.Parameter(weight.detach().clone())   # full, per-layer
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(out_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.normal_(self.A, std=0.01)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        delta = self.scale * (self.A.float() @ self.B.float())
        weight = self.weight + delta.to(self.weight.dtype)
        return F.linear(x, weight, self.bias)


class ConvergenceQwen(nn.Module):
    def __init__(self, model_name="Qwen/Qwen1.5-0.5B", rank=8, alpha=16.0,
                 dtype=torch.float32):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype)
        self.config = self.base.config
        self.rank, self.alpha = rank, alpha
        layers = self.base.model.layers
        self.num_layers = len(layers)

        # Replace each Linear with an AdaptedLinear initialised to the TEACHER's
        # actual per-layer weight (not the mean).
        for i in range(self.num_layers):
            for cls in WEIGHT_CLASSES:
                orig = self._get_linear(layers[i], cls)
                bias = orig.bias if orig.bias is not None else None
                al = AdaptedLinear(orig.weight.data, rank, alpha, bias)
                al.to(device=orig.weight.device, dtype=orig.weight.dtype)
                self._set_linear(layers[i], cls, al)

        # Freeze embeddings / norms / lm_head; train weights + adapters + biases.
        for p in self.base.parameters():
            p.requires_grad = False
        for layer in self.base.model.layers:
            for cls in WEIGHT_CLASSES:
                al = self._get_linear(layer, cls)
                al.weight.requires_grad = True
                al.A.requires_grad = True
                al.B.requires_grad = True
                if al.bias is not None:
                    al.bias.requires_grad = True

    @staticmethod
    def _owner(layer, cls):
        return layer.self_attn if cls in _ATTN_CLASSES else layer.mlp
    def _get_linear(self, layer, cls):
        return getattr(self._owner(layer, cls), cls)
    def _set_linear(self, layer, cls, m):
        setattr(self._owner(layer, cls), cls, m)

    def forward(self, *a, **k):
        return self.base(*a, **k)
    def gradient_checkpointing_enable(self, **k):
        self.base.gradient_checkpointing_enable(**k)

    def class_weights(self, cls):
        """Stack the 24 live per-layer weights of a class -> (24, out, in)."""
        return torch.stack([self._get_linear(l, cls).weight
                            for l in self.base.model.layers], dim=0)

    def consolidation_penalty(self):
        """Sum over classes of mean_i ||W_i - mean_c||^2. Differentiable; the
        mean is part of the graph so the penalty pulls every layer AND the mean
        toward each other (i.e. toward the centroid)."""
        total = 0.0
        for cls in WEIGHT_CLASSES:
            W = self.class_weights(cls)                 # (L, out, in)
            mean = W.mean(dim=0, keepdim=True)          # (1, out, in)
            total = total + ((W - mean) ** 2).mean()
        return total

    @torch.no_grad()
    def spread_per_class(self):
        """Diagnostic (not in the graph): mean L2 distance of each layer's
        weight from its class mean. This is THE metric — its curve over training
        tells you how far convergence reaches."""
        out = {}
        for cls in WEIGHT_CLASSES:
            W = self.class_weights(cls)
            mean = W.mean(dim=0, keepdim=True)
            out[cls] = (W - mean).norm(dim=(1, 2)).mean().item()
        return out

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
    def num_trainable(self):
        return sum(p.numel() for p in self.trainable_parameters())
