"""
Invariant tests for the consolidation logic — fast, CPU-only, no model download.

Builds a *tiny* random Qwen2 (a few small layers) and injects it into
ConsolidatedQwen, so CI can guard the linchpin properties on every push:

  * adapter delta is exactly zero at init (student starts at the per-class mean)
  * each backbone equals the mean of its class's matrices
  * the shared backbone is owned/registered exactly once (ParameterDict)
  * embeddings / final norm / lm_head are frozen; backbones + adapters train
  * gradients actually reach the shared backbone
"""

import copy

import torch
import pytest
from transformers import Qwen2Config, Qwen2ForCausalLM

from model import ConsolidatedQwen, WEIGHT_CLASSES


def tiny_base():
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg)


def build(rank=4):
    return ConsolidatedQwen(rank=rank, base_model=tiny_base())


def test_adapter_zero_at_init():
    """B == 0 so every effective weight == backbone at step 0."""
    m = build()
    for layer in m.base.model.layers:
        for cls in WEIGHT_CLASSES:
            cl = m._get_linear(layer, cls)
            assert torch.count_nonzero(cl.B) == 0
            assert (cl.A @ cl.B).abs().max().item() == 0.0


def test_backbone_is_class_mean():
    base = tiny_base()
    ref = copy.deepcopy(base)            # keep the originals before wrapping
    m = ConsolidatedQwen(rank=4, base_model=base)
    for cls in WEIGHT_CLASSES:
        mats = [getattr(ConsolidatedQwen._owner(layer, cls), cls).weight
                for layer in ref.model.layers]
        mean = torch.stack(mats, dim=0).mean(dim=0)
        assert torch.allclose(m.backbones[cls], mean, atol=1e-6)


def test_backbone_registered_exactly_once():
    """The child ConsolidatedLinear must NOT register the backbone, and each
    backbone object appears exactly once across m.parameters()."""
    m = build()
    cl = m._get_linear(m.base.model.layers[0], "q_proj")
    assert "backbone" not in dict(cl.named_parameters())
    ids = [id(p) for p in m.parameters()]
    for cls in WEIGHT_CLASSES:
        assert ids.count(id(m.backbones[cls])) == 1


def test_freezing():
    m = build()
    assert not m.base.model.embed_tokens.weight.requires_grad
    assert not m.base.model.norm.weight.requires_grad
    assert not m.base.lm_head.weight.requires_grad         # tied -> embed tensor
    for cls in WEIGHT_CLASSES:
        assert m.backbones[cls].requires_grad
    cl = m._get_linear(m.base.model.layers[0], "q_proj")
    assert cl.A.requires_grad and cl.B.requires_grad
    assert cl.bias is not None and cl.bias.requires_grad   # q/k/v carry a bias


def test_forward_runs_and_is_finite():
    m = build()
    ids = torch.randint(0, 256, (2, 16))
    out = m(ids)
    assert out.logits.shape == (2, 16, 256)
    assert torch.isfinite(out.logits).all()


def test_gradients_reach_backbone():
    """After unfreezing, a backward must populate every backbone's grad. (A's
    grad is structurally zero while B==0 — expected LoRA cold-start; B gets the
    first signal. That's exactly why distill.py warms the adapters up first.)"""
    m = build()
    m.set_backbone_requires_grad(True)
    ids = torch.randint(0, 256, (2, 16))
    m(ids, labels=ids).loss.backward()
    for cls in WEIGHT_CLASSES:
        assert m.backbones[cls].grad is not None
    cl = m._get_linear(m.base.model.layers[0], "q_proj")
    assert cl.B.grad is not None


def test_divergence_zero_at_init():
    """B == 0 so every per-layer divergence from the shared mean is exactly 0."""
    m = build()
    div = m.divergence_from_mean()
    for cls in WEIGHT_CLASSES:
        assert len(div[cls]["abs"]) == m.num_layers
        assert all(v == 0.0 for v in div[cls]["abs"])
        assert all(v == 0.0 for v in div[cls]["rel"])


def test_divergence_tracks_active_adapter():
    """Activating one layer's adapter shows up as divergence on that layer only."""
    m = build()
    cl = m._get_linear(m.base.model.layers[0], "q_proj")
    cl.B.data.fill_(0.1)
    div = m.divergence_from_mean()
    assert div["q_proj"]["abs"][0] > 0.0          # layer 0 now diverges
    assert div["q_proj"]["abs"][1] == 0.0         # others untouched


def test_param_groups_split_and_lr():
    m = build()
    groups = m.param_groups(base_lr=1e-3, backbone_lr_mult=0.1)
    by_name = {g["name"]: g for g in groups}
    assert by_name["adapters"]["lr"] == pytest.approx(1e-3)
    assert by_name["backbones"]["lr"] == pytest.approx(1e-4)
    # the 7 backbones live only in the backbone group
    backbone_ids = {id(p) for p in by_name["backbones"]["params"]}
    assert len(backbone_ids) == len(WEIGHT_CLASSES)
    for p in by_name["adapters"]["params"]:
        assert id(p) not in backbone_ids
