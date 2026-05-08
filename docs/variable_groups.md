# Per-Group Prediction Heads (`variable_groups`)

## Overview

By default every channel in a stream shares one `EnsPredictionHead` (MLP trunk + final linear
layer) and one `TargetPredictionEngine` (cross-attention decoder). The `variable_groups` feature
provides two modes of per-group customisation:

| Mode | What differs per group | When to use |
|---|---|---|
| **Option A** | Final projection + activation only | Per-group activations (e.g. Softplus for precipitation); minimal parameter overhead |
| **Option B** | Full cross-attention decoder stack + final projection | Groups that benefit from their own latent routing (e.g. precipitation vs. dynamics) |

Modes can be **mixed within one stream**: some groups can have their own decoder (Option B) while
others share the stream-level decoder (Option A).

The primary use-case is precipitation: `tp` and `cp` must be non-negative, so their head uses
`Softplus`. Option B additionally lets that group run a shallower or differently configured decoder
than the rest of the stream.

---

## Configuration

Add a `variable_groups` key inside a stream config file (e.g. `config/streams/era5_1deg/era5.yml`).

### Option A — shared decoder, per-group heads

```yaml
variable_groups:
  precipitation:
    variables: ["tp", "cp"]          # Python regexes matched with re.fullmatch
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Softplus     # non-negative output

  upper_air:
    variables: ["u_\\d+", "v_\\d+", "z_\\d+", "t_\\d+", "q_\\d+"]
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity

  _default:                          # catches every unmatched channel
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity
```

### Option B — per-group decoder + head

Add `target_readout` inside the group to give it its own cross-attention decoder stack.

```yaml
variable_groups:
  precipitation:
    variables: ["tp", "cp"]
    target_readout:                  # presence of this key triggers Option B for this group
      num_layers: 1
      num_heads: 4
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Softplus

  upper_air:
    variables: ["u_\\d+", "v_\\d+", "z_\\d+", "t_\\d+", "q_\\d+"]
    target_readout:
      num_layers: 3
      num_heads: 8
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity

  _default:
    # No target_readout → shares the stream-level decoder (Option A behaviour)
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity
```

### Config rules

| Rule | Detail |
|---|---|
| `variable_groups` absent | Existing single-head behaviour — fully backward-compatible |
| `variables` | Python regexes matched with `re.fullmatch` against channel names like `"u_850"`, `"tp"` |
| Groups are mutually exclusive | A channel matched by two groups raises `ValueError` at startup |
| `_default` required when any channel is unmatched | `ValueError` raised if omitted |
| `_default` with no unmatched channels | Allowed; the group will have zero channels |
| `target_readout` absent in a group | Group shares the stream-level `TargetPredictionEngine` (Option A) |
| `target_readout` present in a group | Group gets its own `TargetPredictionEngine` (Option B) |
| `decoder_type = Linear` + `target_readout` | Not supported; raises `ValueError` |
| Stream-level TTE | Created only when at least one group lacks `target_readout`; skipped otherwise |

### Available activations

See `src/weathergen/model/utils.py::ActivationFactory._registry` for the full list.

| Name | Use-case |
|---|---|
| `Identity` | Default; no constraint on output range |
| `Softplus` | Non-negative outputs (precipitation, humidity) |
| `Sigmoid` | Outputs in (0, 1) |
| `Tanh` | Outputs in (–1, 1) |
| `GELU` | Smooth nonlinearity |

---

## How it works internally

### 1. Channel names flow into `Model`

`MultiStreamDataSampler.get_target_channels()` returns a `list[list[str]]` (one per stream).
`get_model()` in `model_interface.py` passes this as `targets_channels` to `Model.__init__`.

### 2. Group resolver (`_resolve_variable_groups`)

Defined in `src/weathergen/model/utils.py`. Called during `Model.create()` when `variable_groups`
is present in a stream config.

Returns a list of `(group_name, sorted_channel_indices, group_cfg)` tuples. Channel indices are
positions in `dataset.target_channels` — the same ordering the loss module sees.

### 3. TTE creation in `Model.create()`

For each stream with `variable_groups`:

- **Shared TTE** (`target_token_engines[stream_name]`) — created only when at least one group
  lacks `target_readout` (Option A groups need it).
- **Group TTE** (`target_token_engines["{stream_name}/{group_name}"]`) — created for each group
  that carries `target_readout`. Uses the group's own `num_layers` and `num_heads`.

Stream-level parameters (`dim_head_proj`, `mlp_hidden_factor`, `softcap`) are inherited from the
stream config for all group TTEs.

### 4. Per-group `EnsPredictionHead` instances

One `EnsPredictionHead` is created per group, stored as `pred_heads["{stream_name}/{group_name}"]`.
Each head's `dim_out` equals the number of channels in its group. The input `dim_embed` is the same
for all groups (set by the stream-level `embed_target_coords.dim_embed`).

### 5. `predict_decoders` routing

When `stream_name in model.stream_groups`:

1. Coord embedding runs **once** → `tc_tokens_init [N, dim_embed]`.
2. For each group:
   - **Option B group** (`has_own_tte=True`): `tc_tokens_grp = group_tte(output=tc_tokens_init, ...)`.
   - **Option A group** (`has_own_tte=False`): shared TTE runs once (cached), reused for all such groups.
3. `grp_pred = pred_head(tc_tokens_grp)` → `[ens_size, N, n_group_ch]`.
4. Results scattered into `pred [ens_size, N, total_channels]` at the group's channel indices.
5. Loss module sees the full tensor — **no changes needed in the loss module**.

### 6. Metadata

`model.stream_groups[stream_name]` is a list of `(group_name, ch_indices, has_own_tte)` tuples,
built during `create()` and used by `predict_decoders()` to route each group correctly.

---

## Testing

### Unit tests (CPU, no GPU required)

```bash
# _resolve_variable_groups — index assignment, overlap detection, _default behaviour,
# target_readout pass-through, Option A/B detection logic
uv run pytest src/weathergen/model/variable_groups_test.py -v
```

### GPU tests (EnsPredictionHead shapes + Softplus routing)

```bash
uv run pytest src/weathergen/model/engines_test.py -v
```

### Full unit-test suite

```bash
./scripts/actions.sh unit-test
# engines_test.py is automatically skipped in CPU mode (flash_attn unavailable)
```

### End-to-end integration test

**Option A baseline** — add a single group covering all channels with `Identity` activation;
loss values should match the no-groups baseline exactly (same computation).

**Option B smoke test** — add `target_readout` to one group, verify training starts and loss
decreases normally:

```yaml
# integration_tests/streams/era5_small.yml
variable_groups:
  precip:
    variables: ["tp"]
    target_readout:
      num_layers: 1
      num_heads: 4
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Softplus
  _default:
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity
```

```bash
./scripts/actions.sh integration-test-single
```

---

## Checkpoint compatibility

When `variable_groups` is used:
- `pred_heads` keys change from `"{stream_name}"` → `"{stream_name}/{group_name}"`.
- `target_token_engines` gains extra keys `"{stream_name}/{group_name}"` for Option B groups,
  and may lose `"{stream_name}"` if all groups have their own TTE.

Old checkpoints (without variable groups) cannot be loaded directly into a model with groups.
For new training runs this is not an issue. Resuming an existing run with groups added requires a
manual key remapping (utility not yet implemented).
