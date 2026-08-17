# IMERG diagnostic-decoder finetunes — parent checkpoint comparison

Comparison of the seven pretrained checkpoints used as parents for the IMERG precipitation
decoder finetunes. **`af90zz71` is the reference**; everything else is described relative to it.

All figures were read directly from each checkpoint's saved `model_<run>_*.json` and from the
`.chkpt` tensor shapes — nothing here is inferred from naming conventions.

| | checkpoint read |
|---|---|
| af90zz71 | `af90zz71_latest.chkpt` |
| fx276yn3 | `fx276yn3_chkpt00008.chkpt` |
| cw6a4szu | `cw6a4szu_chkpt00008.chkpt` |
| j6nb50v9 | `j6nb50v9_latest.chkpt` |
| ra1xax01 | `ra1xax01_latest.chkpt` |
| oywmz4sz | `oywmz4sz_chkpt00002.chkpt` (epoch 2, as requested) |
| dy0jlrmw | `dy0jlrmw_latest.chkpt` |

---

## 1. At a glance

| | **af90zz71** *(ref)* | fx276yn3 | cw6a4szu | j6nb50v9 | ra1xax01 | oywmz4sz | dy0jlrmw |
|---|---|---|---|---|---|---|---|
| **Total params** | 1162.8 M | 1180.9 M | 1180.9 M | 1050.1 M | 1050.1 M | 1052.4 M | 1052.4 M |
| **Architecture family** | JEPA/SSL | JEPA/SSL | JEPA/SSL | MTM | MTM | MTM | MTM |
| **Analysis grid** | **o96** | n320 | n320 | n320 | n320 | n320 | n320 |
| **Input channels** | 126 | **327** | 134 | 148 | 148 | 148 | 148 |
| **Native rollout** | 8 steps | 8 | 8 | 8 | 8 | 8 | 8 |
| **Own epochs × samples** | 4 × 8192 | 8 × 8192 | 8 × 8192 | 16 × 8192 | 8 × 8192 | 24 × 16384 | 24 × 16384 |
| **istep at checkpoint** | 5072 | 2048 | 2048 | 6870 | 3822 | 1536 | 5608 |
| **Peak LR** | 3.5e-5 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 5e-5 | 4e-5 |

---

## 2. Model size

Parameter counts by module, in millions.

| | total | encoder | forecast engine | decoder¹ |
|---|---|---|---|---|
| **af90zz71** *(ref)* | **1162.8** | 613.2 | 537.0 | 12.7 |
| fx276yn3 | 1180.9 | 616.2 | 537.1 | 27.6 |
| cw6a4szu | 1180.9 | 616.2 | 537.1 | 27.6 |
| j6nb50v9 | 1050.1 | 487.6 | 537.0 | 25.5 |
| ra1xax01 | 1050.1 | 487.6 | 537.0 | 25.5 |
| oywmz4sz | 1052.4 | 487.6 | 537.1 | 27.6 |
| dy0jlrmw | 1052.4 | 487.6 | 537.1 | 27.6 |

¹ `embed_target_coords` + `target_token_engines` + `pred_heads`.

**The forecast engine is identical across all seven (~537 M).** The size differences come entirely
from the encoder: the JEPA family carries ~613–616 M against the MTM family's 487.6 M, a **126 M
gap**. That is the cost of 64 register tokens plus cross-stream attention, not of depth — see §3.

Decoder size tracks the output stream's channel count, not the backbone: af90zz71 decodes 68
o96 channels (12.7 M) while the n320 models decode 81 (25.5–27.6 M).

---

## 3. Architecture differences

| | af90zz71 *(ref)* | fx276yn3 / cw6a4szu | j6nb50v9 / ra1xax01 / oywmz4sz / dy0jlrmw |
|---|---|---|---|
| `ae_local` dim / blocks | 1024 / 4 | 1024 / 4 | **2048 / 2** |
| `ae_global` dim / blocks | 2048 / 5 | 2048 / 5 | 2048 / **4** |
| `fe_num_blocks` | 16 | 16 | 16 |
| `num_register_tokens` | **64** | 64 | **0** |
| `use_xsa` | ✔ | ✔ | ✘ |
| `with_step_conditioning` | ✔ | ✔ | ✘ |
| `embed_orientation` | channels | channels | channels |
| `decoder_type` | PerceiverIOCoordConditioning | same | same |
| `healpix_level` | 5 | 5 | 5 |
| `with_fsdp` | ✘ | ✘ | **✔** |
| `multiprocessing_method` | spawn | spawn | **fork** |

Two genuinely distinct architectures:

- **JEPA/SSL** (af90zz71, fx276yn3, cw6a4szu) — wide-and-deep local encoder (1024×4), 5 global
  blocks, 64 register tokens, cross-stream attention and step conditioning enabled.
- **MTM** (the remaining four) — a shallower, wider local encoder (2048×2), 4 global blocks, no
  register tokens, no cross-stream attention, no step conditioning. Runs under FSDP.

**`j6nb50v9`/`ra1xax01` and `oywmz4sz`/`dy0jlrmw` are architecturally identical.** They differ
only in lineage and training schedule (§5).

---

## 4. Inputs and outputs

Every one of the seven has **exactly one output stream: `ERA5`** — the diagnostic analysis
decoder. None of them predicts precipitation; IMERG is added fresh by the finetune configs, which
neutralise or drop `ERA5` so that IMERG becomes the sole target.

### Output

| | output stream | channels | grid |
|---|---|---|---|
| **af90zz71** *(ref)* | ERA5 | 68 | o96 |
| all six others | ERA5 | 81 | n320 |

### Inputs (forcing streams)

| stream | af90zz71 *(ref)* | fx276yn3 | cw6a4szu | MTM ×4 |
|---|---|---|---|---|
| ERA5_in (analysis) | 68 (o96) | 74 | 74 | 88 |
| METEOSAT_SEVIRI_IR | 11 | 11 | 11 | 11 |
| GOES_ABI_IR / VIS | 8 / 2 | 8 / 2 | 8 / 2 | 8 / 2 |
| HIMAWARI_AHI_IR / VIS | 8 / 2 | 8 / 2 | 8 / 2 | 8 / 2 |
| AVHRR | — (combined) | 5 | 5 | 5 |
| IASI | METOP_ABC_AVHRR_IASI **18** | METOP_IASI_PC **210** | METOP_ABC_IASI **17** | METOP_ABC_IASI **17** |
| SurfaceCombined | 9 | 7 | 7 | 7 |
| **total input channels** | **126** | **327** | **134** | **148** |

**`fx276yn3` is the outlier: 327 input channels against 134–148 elsewhere.** The entire
difference is its IASI stream — `METOP_IASI_PC` supplies **210 principal components** from the
od-ai BUFR store, where every other model uses `METOP_ABC_IASI` with 17–18 selected radiances.

This has a direct operational consequence: fx276yn3 finetunes need `num_workers: 6` and OOM at 8,
while the others run fine at 8. The ceiling is set by that 12× channel difference, not by the
backbone. See runs `to4xs0gh` (fx276yn3, 8 workers, killed) vs `r2iy7iko` (cw6a4szu, 8 workers,
fine).

> **Unresolved:** `oywmz4sz` was described as the OperAn variant and `dy0jlrmw` as retaining an
> ERA5 initial condition, but their stored configs are identical on this point. Both use
> `ERA5_in` of type `anemoi_operan` reading the **ERA5** store
> `aifs-ea-an-oper-0001-mars-n320-1979-2024-1h-v2-with-era51.zarr`, and for `oywmz4sz` that holds
> at epochs 0, 2, 9 and latest. As configured, the two differ only in parent weights — not in
> initial condition.

---

## 5. Training history

| | lineage | own epochs | samples/epoch | istep | LR peak | grad clip | losses | dates |
|---|---|---|---|---|---|---|---|---|
| **af90zz71** *(ref)* | n0t6ejuo → srdrwfy6 → af90zz71 | 4 | 8192 | 5072 | 3.5e-5 | 0.8 | student-teacher + forecast | 1980–2022 |
| fx276yn3 | si0krkxk → ci2t558k → wejf33qv → fx276yn3 | 8 | 8192 | 2048 | 1e-4 | 1.0 | physical + student-teacher + forecast | 1980–2022 |
| cw6a4szu | wduc34ev → rrvj7vjv → cw6a4szu | 8 | 8192 | 2048 | 1e-4 | 1.0 | physical + student-teacher + forecast | 1980–2022 |
| j6nb50v9 | od08us1u → ccf5i0sc → hsq7hsb6 → j6nb50v9 | 16 | 8192 | 6870 | 1e-4 | 1.0 | physical | 1979–2022 |
| ra1xax01 | od08us1u → ccf5i0sc → hsq7hsb6 → ra1xax01 | 8 | 8192 | 3822 | 1e-4 | 1.0 | physical | 1979–2022 |
| oywmz4sz | skkbxfya → oywmz4sz | 24 (read at **ep 2**) | 16384 | 1536 | 5e-5 | 1.0 | physical | 1979–2022 |
| dy0jlrmw | skkbxfya → dy0jlrmw | 24 | 16384 | 5608 | 4e-5 | 1.0 | physical | 1979–2022 |

Notes:

- **Epoch counts are not comparable across families.** `oywmz4sz`/`dy0jlrmw` use 16384
  samples/epoch, everyone else 8192, so one of their epochs is two of the others'.
- `j6nb50v9` and `ra1xax01` share the ancestor `hsq7hsb6`; `oywmz4sz` and `dy0jlrmw` share
  `skkbxfya`. Within each pair the backbone is the same and only the finetuning differs
  (`ra1xax01` is the "align noise" variant, `ft8_8ep_noise2e-5_baseBetas`).
- The JEPA family carries a `student-teacher` loss block; the MTM family has `physical` only.
  This matters for the finetune configs — see §6.
- **All seven have a native rollout of 8 steps** with a *trainable* forecast engine (no
  forecast-engine term in any freeze regex). An 8-step IMERG finetune therefore asks none of them
  for anything they have not already done, and needs neither a curriculum nor an unfreeze.

---

## 6. Consequences for the finetune configs

The three families look interchangeable in their configs but merge **differently**, which
determines how the ERA5 output stream is removed:

| family | `streams_directory` | `reconstruct: false` | ERA5 removed by | stream dir holds |
|---|---|---|---|---|
| af90zz71 / fx276yn3 / cw6a4szu | unions | supported | `era5.yml` neutraliser | output side only |
| j6nb50v9 / ra1xax01 | **replaces** | **not supported** | simply omitting it | **all 10 streams** |
| oywmz4sz / dy0jlrmw | unions | supported | `era5.yml` neutraliser | output side only |

Other per-family requirements:

| | loss key for MSE | `--mini-epoch` | `num_workers` |
|---|---|---|---|
| af90zz71 / fx276yn3 / cw6a4szu | `forecast` | **required** (no `_latest`) | 6 (fx276yn3) / 8 (cw6a4szu) |
| j6nb50v9 / ra1xax01 | **`physical`** | not needed | 8 |
| oywmz4sz | `physical` | **2** (as requested) | 8 |
| dy0jlrmw | `physical` | not needed | 8 |

The loss key matters: `validation_io.write_output` asserts exactly one `LossPhysical` block.
Declaring the MSE loss under `forecast` in an MTM-family config would sit beside the inherited
`physical` block and trip that assert.

Freeze regex differs too. For the JEPA family the decoder to be *trained* is ERA5-owned, so
`.*ERA5.*` must be **dropped**; for the MTM family IMERG is a new stream, so `.*ERA5.*` is
**kept** and freezes the leftover ERA5 decoder. In both cases a forecast-engine term is added,
leaving only IMERG's fresh decoder trainable (~131 tensors).
