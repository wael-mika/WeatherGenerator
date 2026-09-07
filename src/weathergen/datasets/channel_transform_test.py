# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The per-channel `tp` transform: round-trip fidelity and the positivity guarantee."""

import numpy as np

from weathergen.datasets.data_reader_base import DataReaderBase

SPEC = {"type": "log1p", "scale": 0.1, "mean": 0.3, "stdev": 0.8}


def test_transform_round_trips_on_real_looking_rain():
    """Forward then inverse must return the original field.

    If this drifts, every physical number downstream drifts with it silently -- the transform sits
    between the model and every metric that is quoted in physical units.
    """
    rng = np.random.default_rng(0)
    # rain-like: a large dry atom plus a heavy tail
    x = np.where(rng.random(20000) < 0.7, 0.0, rng.gamma(0.6, 3.0, 20000))
    data = x.copy()[:, None]

    z = DataReaderBase._channel_transform(data.copy(), {0: SPEC}, inverse=False)
    back = DataReaderBase._channel_transform(z, {0: SPEC}, inverse=True)

    assert np.allclose(back[:, 0], x, rtol=1e-5, atol=1e-6)


def test_inverse_can_never_emit_negative_rain():
    """The whole point. Every arm measured so far puts 21-64% of `tp` below zero."""
    rng = np.random.default_rng(1)
    # deliberately wild model output, far outside anything training would produce
    z = rng.normal(0, 5, (50000, 1))
    out = DataReaderBase._channel_transform(z, {0: SPEC}, inverse=True)

    assert (out >= 0).all(), f"min was {out.min()}"
    assert (out == 0).any(), "values below the floor must land exactly on zero, not near it"


def test_transform_compresses_the_tail_it_is_meant_to():
    """It must actually make the marginal easier for a Gaussian-source flow to hit."""
    rng = np.random.default_rng(2)
    x = np.where(rng.random(50000) < 0.7, 0.0, rng.gamma(0.6, 3.0, 50000))[:, None]
    z = DataReaderBase._channel_transform(x.copy(), {0: SPEC}, inverse=False)

    def skew(v):
        v = v[np.isfinite(v)]
        return float(((v - v.mean()) ** 3).mean() / v.std() ** 3)

    assert abs(skew(z[:, 0])) < abs(skew(x[:, 0])), "transform must reduce skew, not add to it"


def test_untransformed_channels_are_untouched():
    """A spec for one channel must not perturb its neighbours."""
    rng = np.random.default_rng(3)
    data = rng.normal(size=(100, 3))
    before = data.copy()
    out = DataReaderBase._channel_transform(data, {1: SPEC}, inverse=False)

    assert np.array_equal(out[:, 0], before[:, 0])
    assert np.array_equal(out[:, 2], before[:, 2])
    assert not np.array_equal(out[:, 1], before[:, 1])


def test_transform_works_on_torch_tensors_too():
    """The inverse runs on PREDICTIONS, which arrive as torch tensors -- on the GPU.

    `validation_io.write_output` calls `denormalize_target_channels` on the model's output, so a
    numpy-only implementation raises "can't convert cuda:N device type tensor to numpy" and kills
    the run at its first validation. That is exactly how arm T1 (jscvtzts) died, silently, after
    one mini-epoch while its matched control ran to completion.
    """
    import torch

    x = torch.rand(500, 2) * 5.0
    want = DataReaderBase._channel_transform(x.numpy().copy(), {0: SPEC}, inverse=False)
    got = DataReaderBase._channel_transform(x.clone(), {0: SPEC}, inverse=False)

    assert torch.is_tensor(got), "a torch input must stay a torch tensor"
    assert np.allclose(got.numpy(), want, atol=1e-5)

    back = DataReaderBase._channel_transform(got, {0: SPEC}, inverse=True)
    assert torch.allclose(back[:, 0], x[:, 0], atol=1e-4)
    assert (back[:, 0] >= 0).all()
