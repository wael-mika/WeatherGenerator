# Combined variable_groups Configuration

This document explains how to use `variable_groups` in a stream config when you want **both**
per-group masking strategies (from [advanced_masking.md](advanced_masking.md)) and per-group
decoder heads / decoders (from [decoder_groups_guide.md](decoder_groups_guide.md)) active at the
same time.

---

## Are the two features compatible?

Yes — there is no code conflict. Each subsystem reads only its own keys from each group entry and
silently ignores the rest:

| Subsystem | Keys it reads | Keys it ignores |
|---|---|---|
| `masking.py` (spatial mask generation) | `masking`, `masking_rate` | `pred_head`, `target_readout` |
| `tokenizer_masking.py` (2-D channel mask) | `variables` | everything else |
| `model/utils.py` `_resolve_variable_groups` | `variables`, `pred_head`, `target_readout` | `masking`, `masking_rate` |
| `model/model.py` (decoder build) | `pred_head`, `target_readout` | `masking`, `masking_rate` |

A single group entry can carry all four families of keys simultaneously.

---

## Four rules to avoid silent failures

1. **Every named group must have a `variables` key** (except `_default`).  
   Both masking and decoder code use `variables` for channel-to-group assignment. A group that
   exists only to carry masking config but omits `variables` will silently produce zero channel
   coverage for the decoder.

2. **Every group must have `pred_head`, or the stream must have a top-level `pred_head` fallback**.  
   `model/model.py` falls back to the stream-level key; if neither exists at startup it raises
   `omegaconf.errors.ConfigKeyError`.

3. **Group names must be identical everywhere they appear**.  
   - Stream config `variable_groups` keys (the authoritative definition)  
   - Any `variable_groups: [name1, name2]` tags inside `model_input` entries in the training config
     (Mode B masking)  
   A typo causes the Mode B 2-D channel mask to silently default to "all channels visible" for
   the misnamed group. The training code now raises `ValueError` at startup when Mode B group
   names are not found in the stream config.

4. **Do not mix Mode A and Mode B masking for the same stream**.  
   - Mode A: groups have `masking` or `masking_rate` keys in the stream config.  
   - Mode B: `model_input` entries carry `variable_groups: [...]` tags in the training config.  
   Mode A takes priority. If both are present the training code emits a `WARNING` and Mode B
   tags are ignored. Use one or the other per stream.

---

## Complete combined template

The template below shows a single ERA5 stream configured with per-group masking (Mode A) and
per-group decoder heads (Option A — shared TTE, per-group head). Inline comments identify which
feature each key belongs to.

### Stream config (`config/streams/era5_1deg_combined/era5.yml`)

```yaml
ERA5:
  type: anemoi
  filenames: ['aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr']
  stream_id: 0

  source_exclude: ['w_', 'skt', 'tcw', 'cp', 'tp']
  target_exclude: ['w_', 'slor', 'sdor', 'tcw']

  loss_weight: 1.0
  location_weight: cosine_latitude
  token_size: 8
  tokenize_spacetime: True
  max_num_targets: -1

  # Optional: drop individual channels with low probability (source-side only).
  # See advanced_masking.md § Channel Dropout.
  channel_drop_rate: 0.005      # decoder_groups feature — not part of variable_groups

  embed:
    net: transformer
    num_tokens: 1
    num_heads: 8
    dim_embed: 256
    num_blocks: 2

  embed_target_coords:
    net: linear
    dim_embed: 256

  # Stream-level fallback head — used by any group that omits pred_head.
  # decoder_groups feature.
  pred_head:
    ens_size: 1
    num_layers: 1
    final_activation: Identity

  # Shared cross-attention decoder for all groups (Option A).
  # decoder_groups feature. Required when any group has no target_readout.
  target_readout:
    num_layers: 2
    num_heads: 4

  variable_groups:

    precipitation:
      # ── Channel assignment (required by BOTH subsystems) ──────────────────
      variables: ["tp", "cp"]

      # ── Per-group masking (advanced_masking.md, Mode A) ───────────────────
      masking:
        strategy: satellite_swath
        config:
          num_swaths: 5
          swath_width_deg: 20
          orbit_drift_deg: -25

      # ── Per-group decoder head (decoder_groups_guide.md, Option A) ─────────
      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Softplus    # tp and cp are non-negative

    upper_air:
      variables:
        - "u_\\d+"
        - "v_\\d+"
        - "z_\\d+"
        - "t_\\d+"
        - "q_\\d+"

      masking:
        strategy: random
        config:
          rate: 0.2
          rate_sampling: true
          rate_distribution: beta     # advanced_masking.md § Beta Sampling
          rate_alpha: 2.0

      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity

    _default:
      # _default has no variables key — it catches every unmatched channel automatically.
      masking:
        strategy: healpix
        config:
          rate: 0.25
          hl_mask: 4
          rate_sampling: true
          rate_distribution: beta
          rate_alpha: 3.0

      pred_head:
        ens_size: 1
        num_layers: 1
        final_activation: Identity
```

### Training config (relevant `model_input` excerpt)

When Mode A masking is active (any group has `masking`/`masking_rate` in the stream config),
the `masking_strategy` in `model_input` is **not used for mask generation** — it is overridden
by the per-group masks. The `model_input` entry is still required for the loss correspondence
mapping.

```yaml
training_config:
  model_input:
    mae_source:
      masking_strategy: random    # used only if no variable_groups.masking in stream config
      num_samples: 1
      num_steps_input: 1
      masking_strategy_config:
        rate: 0.2
      # DO NOT add variable_groups: [...] here when Mode A is active in the stream config.
      # Adding both triggers a startup WARNING and Mode B tags are ignored.
```

---

## Adding Option B decoders (per-group TTE)

To give a group its own cross-attention decoder instead of sharing the stream-level one, add
`target_readout` inside the group entry. The stream-level `target_readout` is still required if
any other group is Option A (no per-group `target_readout`).

```yaml
variable_groups:

  precipitation:
    variables: ["tp", "cp"]
    masking:
      strategy: satellite_swath
      config: {num_swaths: 5, swath_width_deg: 20, orbit_drift_deg: -25}
    target_readout:         # Option B: own 1-layer decoder for precipitation
      num_layers: 1
      num_heads: 4
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Softplus

  _default:                 # Option A: shares the stream-level 2-layer decoder
    masking:
      strategy: random
      config: {rate: 0.2}
    pred_head:
      ens_size: 1
      num_layers: 1
      final_activation: Identity
```

---

## Interaction summary

| Feature combination | Outcome |
|---|---|
| Masking (Mode A) + decoder Option A | Fully supported. One `variable_groups` block carries both. |
| Masking (Mode A) + decoder Option B | Fully supported. Add `target_readout` to groups that need their own TTE. |
| Masking (Mode B, model_input tags) + decoder groups | Supported. Group names in `model_input` tags **must** match stream config keys; mismatch now raises `ValueError`. |
| Masking Mode A + Mode B simultaneously | Mode A wins; startup emits `WARNING`. Use only one per stream. |
| Decoder groups only, no masking keys | Default stream-level spatial masking applies; per-group 2-D channel masks are not built. This is the current behaviour of `era5_1deg_physics_optA/`. |
| Channel dropout + any masking/decoder combination | Independent; channel dropout applies on top of whatever spatial mask is in use. |
