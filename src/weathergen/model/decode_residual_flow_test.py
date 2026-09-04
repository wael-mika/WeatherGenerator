# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Integration tests for the residual flow-matching decoder.

These build a real decoder, so they need ``flash_attn`` and a GPU (the varlen attention has no
CPU path) and are skipped otherwise. The algebra the decoder rests on is covered on CPU by
``flow_math_test.py``; what is tested here is the wiring: the training pack, the conditioning
switch, the freeze mechanism, and -- most importantly -- warm-start key compatibility.
"""

import pytest
import torch
from omegaconf import OmegaConf

from weathergen.model.decode_residual_flow import ResidualFlowPointDecoder
from weathergen.model.engines import FlowMatchingPointDecoder, TargetPredictionEngineClassic

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="varlen flash attention needs a GPU"
)

DIM, C, HEADS, DEV = 32, 3, 4, "cuda"
_STREAM = OmegaConf.create({"name": "CERRA", "target_readout": {"num_heads": HEADS}})


def _cf(**over):
    d = dict(
        ae_global_dim_embed=DIM,
        with_flash_attention=True,
        norm_type="LayerNorm",
        norm_eps=1e-5,
        mlp_norm_eps=1e-5,
        attention_dtype="bf16",
        pred_self_attention=True,
        pred_mlp_adaln=True,
        flow_dim_time=8,
        flow_num_steps=3,
        flow_num_layers=2,
    )
    d.update(over)
    return OmegaConf.create(d)


def _args(**over):
    return (_cf(**over), [DIM, DIM], 4, DIM // HEADS, 2, 0.0)


def _build(**over):
    torch.manual_seed(7)
    return ResidualFlowPointDecoder(*_args(**over), stream_config=_STREAM, num_channels=C).to(DEV)


@pytest.fixture
def inputs():
    counts = torch.tensor([3, 5, 4], dtype=torch.int32)
    n, ncell = int(counts.sum()), counts.numel()
    torch.manual_seed(0)
    return dict(
        n=n,
        latent=torch.randn(ncell * 9, DIM, device=DEV),
        latent2=torch.randn(ncell * 9, DIM, device=DEV),
        output=torch.randn(n, DIM, device=DEV),
        coordinates=torch.randn(n, 4, device=DEV),
        target=torch.randn(n, C, device=DEV),
        output_lens=torch.cat([torch.zeros(1, dtype=torch.int32), counts]).to(DEV),
        latent_lens=torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.full((ncell,), 9, dtype=torch.int32)]
        ).to(DEV),
    )


def _decode(dec, io, target=None, ens_size=1, latent=None):
    """Deterministic pass then the corrector, under the autocast the trainer runs in."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        det = dec.det_forward(
            latent=io["latent"],
            output=io["output"],
            latent_lens=io["latent_lens"],
            output_lens=io["output_lens"],
            coordinates=io["coordinates"],
        )
        mu = det[:, :C].float()
        pred = dec.correct(
            mu=mu,
            det_tokens=det,
            coord_tokens=io["output"],
            latent=io["latent"] if latent is None else latent,
            latent_lens=io["latent_lens"],
            output_lens=io["output_lens"],
            coordinates=io["coordinates"],
            target=target,
            ens_size=ens_size,
        )
    return mu, det, pred


def test_training_returns_the_two_member_pack(inputs):
    dec = _build().train()
    mu, _, pred = _decode(dec, inputs, target=inputs["target"])

    assert pred.shape == (3, inputs["n"], C)
    assert torch.allclose(pred[0], mu, atol=1e-5), "member 0 must be mu itself"
    # member 2 is the per-channel r_scale, which mse_flow divides out so that every channel's
    # CFM term carries the same weight regardless of how well the base already fits it
    assert torch.allclose(pred[2], dec.flow.r_scale.value().expand_as(pred[2]).float(), atol=1e-5)
    assert (pred[2] > 0).all()

    pred.float().sum().backward()
    assert dec.flow.vel_head.weight.grad is not None
    assert any(p.grad is not None for p in dec.tte.parameters()), (
        "member 0 is undetached, so mse_det must still reach the deterministic stack"
    )


@pytest.mark.parametrize("freeze_base", [False, True])
def test_training_pass_calibrates_the_residual_scale(inputs, freeze_base):
    """A training pass MUST move ``r_scale`` off the global scalar 1.0.

    This is the test whose absence cost the level-5 campaign. ``ResidualScale`` is correct and
    well covered in isolation by ``flow_math_test.py``, but nothing asserted that the decoder
    ever *calls* it -- and it did not: ``fthi701s`` and ``zk7yuisj`` trained 28 mini-epochs with
    ``count == 0`` and ``scale == [1]*C``, i.e. the single global scalar that
    ``ResidualScale``'s own docstring explains cannot survive this data. The result was a
    corrector injecting 2-5x too much residual amplitude on 7 of 8 channels.
    """
    dec = _build(flow_freeze_base=freeze_base).train()
    assert int(dec.flow.r_scale.count) == 0, "fixture should start uncalibrated"

    # distinct per-channel spread, so a per-channel scale is distinguishable from a scalar one
    target = inputs["target"] * torch.tensor([0.2, 1.0, 5.0], device=DEV)
    _decode(dec, inputs, target=target)

    scale = dec.flow.r_scale.scale
    assert int(dec.flow.r_scale.count) > 0, "the training pass never called r_scale.update()"
    assert not torch.allclose(scale, torch.ones_like(scale)), (
        f"r_scale never moved off the forbidden global scalar 1.0: {scale.tolist()}"
    )
    assert scale[2] > scale[0], (
        f"r_scale is not tracking per-channel residual spread: {scale.tolist()}"
    )


def test_eval_pass_does_not_calibrate(inputs):
    """Sampling must integrate with exactly the scale training settled on, never re-estimate it."""
    dec = _build().eval()
    _decode(dec, inputs, ens_size=2)
    assert int(dec.flow.r_scale.count) == 0


def test_nan_targets_do_not_poison_the_pack(inputs):
    dec = _build().train()
    tgt = inputs["target"].clone()
    tgt[::3, 0] = float("nan")
    tgt[1, :] = float("nan")

    _, _, pred = _decode(dec, inputs, target=tgt)
    assert torch.isfinite(pred).all()


def test_eval_draws_distinct_members(inputs):
    dec = _build().eval()
    _, _, samples = _decode(dec, inputs, ens_size=4)

    assert samples.shape == (4, inputs["n"], C)
    assert not torch.allclose(samples[0], samples[1]), "members must differ in the base noise"


def test_flow_ens_size_zero_emits_bare_mu(inputs):
    """The escape hatch that scores a corrector checkpoint as a purely deterministic model."""
    dec = _build(flow_ens_size=0).eval()
    mu, _, samples = _decode(dec, inputs, ens_size=4)

    assert samples.shape == (1, inputs["n"], C)
    assert torch.allclose(samples[0], mu)


@pytest.mark.parametrize(
    ("cond", "uses_latent"),
    [("mu", False), ("mu+tokens", False), ("mu+tokens+latent", True)],
)
def test_flow_cond_controls_latent_conditioning(inputs, cond, uses_latent):
    """Variant 1 must not read the latent; variant 2 must.

    The comparison holds ``mu`` and the decoder tokens fixed and swaps only the KV, so it isolates
    the corrector's own conditioning from the latent information already baked into ``mu``.
    """
    dec = _build(flow_cond=cond).eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        det = dec.det_forward(
            latent=inputs["latent"],
            output=inputs["output"],
            latent_lens=inputs["latent_lens"],
            output_lens=inputs["output_lens"],
            coordinates=inputs["coordinates"],
        )
        mu = det[:, :C].float()
        kw = dict(
            mu=mu,
            det_tokens=det,
            coord_tokens=inputs["output"],
            latent_lens=inputs["latent_lens"],
            output_lens=inputs["output_lens"],
            coordinates=inputs["coordinates"],
            ens_size=2,
        )
        # seed immediately before each call: anything drawn in between would shift the base noise
        # and make the two samples differ for the wrong reason
        torch.manual_seed(3)
        a = dec.correct(latent=inputs["latent"], **kw)
        torch.manual_seed(3)
        b = dec.correct(latent=inputs["latent2"], **kw)

    assert torch.allclose(a, b) != uses_latent


def test_guidance_requires_latent_conditioning():
    """With a null KV the guidance knobs are silently inert -- fail loudly instead."""
    for over in ({"flow_cond_dropout": 0.1}, {"flow_guidance": 2.0}):
        with pytest.raises(AssertionError, match="flow_cond='mu\\+tokens\\+latent'"):
            _build(flow_cond="mu+tokens", **over)


def test_freeze_base_survives_train_mode():
    """requires_grad_ alone leaves dropout live; a frozen base must also be forced to eval().

    Otherwise the trainer's per-epoch model.train() makes mu stochastic and the corrector spends
    its capacity undoing dropout noise.
    """
    dec = _build(flow_freeze_base=True).train()

    assert dec.flow.training, "the corrector itself must stay in training mode"
    assert not dec.tte.training, "the frozen deterministic stack must be held in eval"
    assert not any(p.requires_grad for p in dec.tte.parameters())


def test_warm_start_from_a_deterministic_parent():
    """Every inherited key must match, so a deterministic checkpoint loads cleanly.

    This is why the class calls super().__init__ with the UN-widened dim_coord_in.
    """
    det = TargetPredictionEngineClassic(*_args(), stream_config=_STREAM)
    dec = ResidualFlowPointDecoder(*_args(), stream_config=_STREAM, num_channels=C)

    missing, unexpected = dec.load_state_dict(det.state_dict(), strict=False)
    assert not unexpected
    assert missing, "the corrector's own weights should be reported missing"
    assert all(k.startswith("flow.") for k in missing), (
        f"inherited keys did not match: {[k for k in missing if not k.startswith('flow.')][:5]}"
    )


def test_every_reinit_root_can_be_reset():
    """load_model_state reinitialises missing keys by calling reset_parameters() on the
    highest-level module covering them. A bare parameter or buffer on the branch would make that
    root a TargetPredictionEngineClassic, which has none, and warm starting would die."""
    det = TargetPredictionEngineClassic(*_args(), stream_config=_STREAM)
    dec = ResidualFlowPointDecoder(*_args(), stream_config=_STREAM, num_channels=C)
    missing, _ = dec.load_state_dict(det.state_dict(), strict=False)

    roots: set[str] = set()
    for path in sorted({k.rsplit(".", 1)[0] for k in missing}):
        if not any(path.startswith(r + ".") for r in roots):
            roots.add(path)

    modules = dict(dec.named_modules())
    for root in roots:
        assert hasattr(modules[root], "reset_parameters"), f"{root} has no reset_parameters"
        modules[root].reset_parameters()


def test_warm_start_from_a_flow_matching_parent_is_rejected():
    """FlowMatching widens the AdaLN aux, so its tte.* weights are the wrong shape.

    strict=False forgives missing/unexpected keys but still raises on a size mismatch, so this
    fails loudly rather than silently training on a half-loaded model.
    """
    fm = FlowMatchingPointDecoder(*_args(), stream_config=_STREAM, num_channels=C)
    dec = ResidualFlowPointDecoder(*_args(), stream_config=_STREAM, num_channels=C)

    with pytest.raises(RuntimeError, match="size mismatch"):
        dec.load_state_dict(fm.state_dict(), strict=False)
