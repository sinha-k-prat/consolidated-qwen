"""
model.py — ConsolidatedQwen

NOVEL IDEA (consolidation + per-class LoRA compression)
=======================================================
A Qwen2 decoder layer contains 7 weight "classes":

    q_proj, k_proj, v_proj, o_proj   (attention)
    gate_proj, up_proj, down_proj    (mlp)

Across the 24 layers of Qwen1.5-0.5B that is 24 independent matrices per class
(168 matrices total). We hypothesise the 24 matrices of a given class cluster
around a common transform, so instead of storing 24 full matrices we store:

    ONE shared backbone matrix per class   (init = mean of the 24 matrices)
  + a per-layer rank-r LoRA adapter        (captures that layer's deviation)

    W_eff[layer=i, class=c] = backbone[c] + scale * (A[i,c] @ B[i,c])

  - backbone[c] : (out, in)  shared by all 24 layers of class c, init = class mean
  - A[i,c]      : (out, r)    per-layer, SMALL-RANDOM init
  - B[i,c]      : (r, in)     per-layer, ZERO init   ->  A @ B = 0 at step 0
  - scale       : alpha / r   (standard LoRA scaling)

Because B starts at zero, every layer's effective weight at step 0 is exactly
the shared backbone (the per-class mean). This is a deliberately lossy starting
point; knowledge distillation then has to *earn back* each layer's individuality
through its cheap rank-r adapter instead of a full matrix.

Storage intuition: 24 full matrices  ->  1 full matrix + 24 * r * (out+in)
cheap adapter factors. For small r the adapters are tiny next to a 1024x1024 or
2816x1024 matrix. NOTE: the frozen embeddings + lm_head dominate the absolute
size budget and do not compress here — evaluate.py reports the full breakdown.

Frozen & full-precision: embeddings, all layernorms, final norm, lm_head.
Trainable: the 7 backbones + every layer's A/B + the q/k/v biases.
"""

from __future__ import annotations   # PEP 604 (X | None) annotations on py<3.10

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# The 7 weight classes inside every Qwen2 decoder layer.
WEIGHT_CLASSES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Which submodule of a decoder layer owns each class.
_ATTN_CLASSES = {"q_proj", "k_proj", "v_proj", "o_proj"}


class ConsolidatedLinear(nn.Module):
    """Drop-in replacement for an nn.Linear inside one decoder layer.

        W_eff = backbone + scale * (A @ B)

    Registered (owned) here: A, B, and the per-layer bias.
    NOT owned here: the shared `backbone`, which is owned exactly once by the
    parent ConsolidatedQwen's `backbones` ParameterDict.

    Why `object.__setattr__` for the backbone reference
    ---------------------------------------------------
    We want the backbone registered EXACTLY ONCE (single owner, single
    state_dict key, one set of optimizer params) while all 24 layers of a class
    read the same tensor object so autograd accumulates their gradients into it.

    PyTorch's nn.Module.__setattr__ intercepts ANY nn.Parameter assigned as a
    normal attribute and re-registers it on that module. So `self.backbone =
    backbone` here would register a *second* copy on every layer (24 extra keys
    in state_dict). There is no plain assignment that both references a
    parent-owned Parameter and avoids re-registration — holding the reference
    requires one minimal escape hatch. `object.__setattr__` stores the reference
    directly in __dict__, bypassing registration. The parameter therefore stays
    owned solely by the parent ParameterDict; gradients still flow because the
    autograd graph is built from the tensor used in forward(), not from where it
    is registered.
    """

    def __init__(self, backbone: nn.Parameter, rank: int, alpha: float,
                 bias: torch.Tensor | None):
        super().__init__()
        out_features, in_features = backbone.shape

        # Reference (not a registration) to the shared, parent-owned backbone.
        object.__setattr__(self, "backbone", backbone)

        self.rank = rank
        self.scale = alpha / rank

        # Per-layer LoRA factors.
        #   A: small random  ->  gives the adapter a non-degenerate direction
        #   B: zeros         ->  A @ B == 0 at init  ->  layer == backbone (mean)
        self.A = nn.Parameter(torch.empty(out_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.normal_(self.A, std=0.01)
        # B is left as zeros on purpose — this is the linchpin of the experiment.

        if bias is not None:
            # Keep the original per-layer bias, full precision, trainable.
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the low-rank delta in fp32 for numerical stability (small A,
        # scale=alpha/r can underflow in fp16), then cast back to the weight
        # dtype before the linear. backbone/A/B share device; dtype may differ.
        delta = self.scale * (self.A.float() @ self.B.float())   # fp32
        weight = self.backbone + delta.to(self.backbone.dtype)
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        out_f, in_f = self.backbone.shape
        return (f"in_features={in_f}, out_features={out_f}, rank={self.rank}, "
                f"scale={self.scale:.3f}, bias={self.bias is not None}")


class ConsolidatedQwen(nn.Module):
    """Qwen1.5-0.5B with all 7 weight classes consolidated into shared
    backbones + per-layer LoRA adapters."""

    def __init__(self, model_name: str = "Qwen/Qwen1.5-0.5B",
                 rank: int = 8, alpha: float = 16.0,
                 dtype: torch.dtype = torch.float32,
                 base_model: nn.Module | None = None):
        super().__init__()
        # `base_model` lets tests inject a tiny random Qwen2 so CI can check the
        # consolidation invariants without downloading the 1 GB checkpoint.
        if base_model is not None:
            self.base = base_model
        else:
            self.base = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype)
        self.config = self.base.config
        self.rank = rank
        self.alpha = alpha

        layers = self.base.model.layers
        self.num_layers = len(layers)

        # ---- 1. Build one backbone per class = MEAN of its 24 matrices -------
        # All 24 matrices of a class share an identical shape, so stacking and
        # averaging is well-defined. This ParameterDict is the SINGLE owner of
        # the shared backbones (clean key: backbones.<class>).
        self.backbones = nn.ParameterDict()
        for cls in WEIGHT_CLASSES:
            mats = [self._get_linear(layers[i], cls).weight.data
                    for i in range(self.num_layers)]
            mean = torch.stack(mats, dim=0).mean(dim=0)   # (out, in)
            self.backbones[cls] = nn.Parameter(mean.clone())

        # ---- 2. Replace every Linear with a ConsolidatedLinear --------------
        # Each new module references its class backbone and adds its own
        # zero-initialized adapter, so the student initially == per-class mean.
        for i in range(self.num_layers):
            for cls in WEIGHT_CLASSES:
                orig = self._get_linear(layers[i], cls)
                bias = orig.bias if orig.bias is not None else None
                cl = ConsolidatedLinear(self.backbones[cls], rank, alpha, bias)
                cl.to(device=orig.weight.device, dtype=orig.weight.dtype)
                self._set_linear(layers[i], cls, cl)

        # ---- 3. Freeze everything that isn't a backbone or adapter ----------
        self._freeze_non_consolidated()

    # --- helpers to locate a class's Linear within a decoder layer -----------
    @staticmethod
    def _owner(layer, cls):
        return layer.self_attn if cls in _ATTN_CLASSES else layer.mlp

    def _get_linear(self, layer, cls):
        return getattr(self._owner(layer, cls), cls)

    def _set_linear(self, layer, cls, new_mod):
        setattr(self._owner(layer, cls), cls, new_mod)

    def _freeze_non_consolidated(self):
        # Freeze ALL base params first (embeddings, layernorms, final norm,
        # lm_head, and the adapter tensors that live under base.model.layers).
        # The shared backbones are NOT here — they live in self.backbones and
        # are never registered on the children (see ConsolidatedLinear docs).
        for p in self.base.parameters():
            p.requires_grad = False
        # Backbones: trainable.
        for cls in WEIGHT_CLASSES:
            self.backbones[cls].requires_grad = True
        # Re-enable per-layer adapters (+ q/k/v biases).
        for layer in self.base.model.layers:
            for cls in WEIGHT_CLASSES:
                cl = self._get_linear(layer, cls)
                cl.A.requires_grad = True
                cl.B.requires_grad = True
                if cl.bias is not None:
                    cl.bias.requires_grad = True

    # --- forward just delegates; supports output_hidden_states for KD --------
    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
        self.base.gradient_checkpointing_enable(**kwargs)

    # --- optimizer parameter groups -----------------------------------------
    def param_groups(self, base_lr: float, backbone_lr_mult: float = 0.1):
        """Two AdamW groups. The backbone is shared across 24 layers, so it
        accumulates ~24x the gradient signal — give it a lower LR so it does not
        drift before the per-layer adapters have established themselves."""
        backbone_params = [self.backbones[c] for c in WEIGHT_CLASSES]
        backbone_ids = {id(p) for p in backbone_params}
        adapter_params = [p for p in self.parameters()
                          if p.requires_grad and id(p) not in backbone_ids]
        return [
            {"params": adapter_params, "lr": base_lr, "name": "adapters"},
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult,
             "name": "backbones"},
        ]

    def set_backbone_requires_grad(self, flag: bool):
        """Used by the backbone warm-up freeze in distill.py."""
        for c in WEIGHT_CLASSES:
            self.backbones[c].requires_grad = flag

    # --- introspection / checkpointing --------------------------------------
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    @torch.no_grad()
    def divergence_from_mean(self):
        """How far each layer's effective weight sits from the shared backbone
        (the consolidated per-class 'mean'), per layer and per class.

        Effective weight = backbone + scale*(A@B). Since the backbone IS the one
        shared/mean component, the per-layer divergence is exactly the adapter
        delta. (This is the V1 analog of V2's per-class spread metric, but read
        from the adapters instead of from separate per-layer weights.)

        Returns { class: {"abs": [L floats], "rel": [L floats]} } where
        abs = ||delta_i||_F  and  rel = ||delta_i||_F / ||backbone||_F.
        """
        out = {}
        for cls in WEIGHT_CLASSES:
            backbone = self.backbones[cls]
            bnorm = backbone.float().norm().item()
            abs_list, rel_list = [], []
            for layer in self.base.model.layers:
                cl = self._get_linear(layer, cls)
                delta = cl.scale * (cl.A.float() @ cl.B.float())
                dn = delta.norm().item()
                abs_list.append(dn)
                rel_list.append(dn / (bnorm + 1e-12))
            out[cls] = {"abs": abs_list, "rel": rel_list}
        return out

    def consolidated_state_dict(self) -> dict:
        """Only the cheap learned tensors (backbones + adapters + biases).
        This is what we save; evaluate.py adds the shared frozen params
        (embeddings/norms/lm_head) back when reporting honest on-disk size."""
        sd = {f"backbone.{c}": self.backbones[c].detach().cpu()
              for c in WEIGHT_CLASSES}
        for i, layer in enumerate(self.base.model.layers):
            for cls in WEIGHT_CLASSES:
                cl = self._get_linear(layer, cls)
                sd[f"layer{i}.{cls}.A"] = cl.A.detach().cpu()
                sd[f"layer{i}.{cls}.B"] = cl.B.detach().cpu()
                if cl.bias is not None:
                    sd[f"layer{i}.{cls}.bias"] = cl.bias.detach().cpu()
        return sd

    def load_consolidated_state_dict(self, sd: dict):
        for c in WEIGHT_CLASSES:
            self.backbones[c].data.copy_(sd[f"backbone.{c}"])
        for i, layer in enumerate(self.base.model.layers):
            for cls in WEIGHT_CLASSES:
                cl = self._get_linear(layer, cls)
                cl.A.data.copy_(sd[f"layer{i}.{cls}.A"])
                cl.B.data.copy_(sd[f"layer{i}.{cls}.B"])
                key = f"layer{i}.{cls}.bias"
                if cl.bias is not None and key in sd:
                    cl.bias.data.copy_(sd[key])


if __name__ == "__main__":
    # Tiny self-check (CPU). Verifies the linchpin: student == per-class mean at
    # init, i.e. every adapter delta is exactly zero.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen1.5-0.5B")
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()

    m = ConsolidatedQwen(args.model_name, rank=args.rank)
    print(m.base.model.layers[0].self_attn.q_proj)
    print(f"tie_word_embeddings = {m.config.tie_word_embeddings}")
    print(f"num decoder layers  = {m.num_layers}")
    print(f"trainable params    = {m.num_trainable():,}")

    # Adapter-zero guarantee: every B is zero, so every delta is zero.
    max_delta = 0.0
    for layer in m.base.model.layers:
        for cls in WEIGHT_CLASSES:
            cl = m._get_linear(layer, cls)
            max_delta = max(max_delta, (cl.A @ cl.B).abs().max().item())
    print(f"max |A@B| at init   = {max_delta:.3e}  (must be 0.0)")
    assert max_delta == 0.0, "adapter delta must be exactly zero at init"
    print("OK: student starts exactly at the per-class mean.")
