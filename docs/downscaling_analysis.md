# WeatherGenerator Downscaling: Architecture Analysis and Implementation Guide

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Model Architecture for Downscaling](#model-architecture-for-downscaling)
3. [HEALPix Tokenization System](#healpix-tokenization-system)
4. [Data Flow Analysis](#data-flow-analysis)
5. [Current Resolution Issue](#current-resolution-issue)
6. [Implementation Fix](#implementation-fix)
7. [Key Files Reference](#key-files-reference)
8. [Memory Considerations](#memory-considerations)

---

## Executive Summary

The WeatherGenerator uses a HEALPix-based transformer architecture for weather prediction and downscaling. For downscaling from ERA5 (~110km) to CERRA (5.5km), the model:

1. **Encodes** ERA5 data into a latent representation organized by HEALPix cells
2. **Processes** the latent through local and global attention mechanisms
3. **Decodes** to CERRA coordinates using a PerceiverIO-style cross-attention decoder

**Key Finding**: The current implementation uses the same HEALPix level for both source (ERA5) and target (CERRA), limiting output resolution.

---

## Model Architecture for Downscaling

### Overview

```
ERA5 Input -> Embedding -> Local Attention -> Global Attention -> PerceiverIO Decoder -> CERRA Output
   (~110km)                  (HEALPix cells)                       (cross-attention)     (5.5km)
```

### Key Components

#### 1. Embedding Engine
- Converts input data into token embeddings
- Groups data points by HEALPix cells
- Each cell becomes one or more tokens

#### 2. Local Assimilation Engine
- Processes tokens within each HEALPix cell
- Captures local spatial relationships
- Uses transformer blocks

#### 3. Global Assimilation Engine
- Models long-range dependencies across cells
- Uses sparse attention patterns (block-sparse)
- Produces the final latent representation

#### 4. Target Prediction Engine (PerceiverIO Decoder)
- Queries the latent representation at specific target coordinates
- Uses cross-attention: targets are queries, latent is key/value
- Resolution-agnostic: can output at any coordinate

### Config Reference
```yaml
# Local autoencoder
ae_local_dim_embed: 128
ae_local_num_blocks: 4
ae_local_num_heads: 8

# Global autoencoder
ae_global_dim_embed: 256
ae_global_num_blocks: 4
ae_global_num_heads: 8

# Decoder type
decoder_type: PerceiverIOCoordConditioning
```

---

## HEALPix Tokenization System

### What is HEALPix?

HEALPix (Hierarchical Equal Area isoLatitude Pixelization) divides the sphere into equal-area cells. The resolution is controlled by a single parameter: the HEALPix level.

### HEALPix Level to Resolution Mapping

| Level | Number of Cells | Approx Cell Size | Suitable Data Resolution |
|-------|-----------------|------------------|--------------------------|
| 3 | 768 | ~2,960 km | Very coarse global |
| 4 | 3,072 | ~1,480 km | Coarse global |
| 5 | 12,288 | ~370 km | ERA5 (~110km) OK |
| 6 | 49,152 | ~185 km | Intermediate |
| 7 | 196,608 | ~92 km | High-res regional |
| 8 | 786,432 | ~46 km | Very high-res |
| 9 | 3,145,728 | ~23 km | Near CERRA resolution |
| 10 | 12,582,912 | ~11 km | Close to CERRA |
| 11 | 50,331,648 | ~5.5 km | Full CERRA resolution |

**Formula**: `num_cells = 12 * 4^level`

### How Tokenization Works

1. **Coordinate Assignment**: Each data point (lat, lon) is assigned to a HEALPix cell
2. **Cell Grouping**: Points are grouped by their cell index
3. **Token Creation**: Each cell's points become tokens (up to `token_size` per cell)
4. **Padding**: Cells with fewer points are padded

**Code location**: `src/weathergen/datasets/tokenizer_utils.py`

```python
def hpy_cell_splits(coords: torch.tensor, hl: int):
    """Compute healpix cell id for each coordinate on given level hl"""
    thetas = ((90.0 - coords[:, 0]) / 180.0) * np.pi
    phis = ((coords[:, 1] + 180.0) / 360.0) * 2.0 * np.pi
    hpy_idxs = ang2pix(2**hl, thetas, phis, nest=True)
    # ... returns indices per cell
```

---

## Data Flow Analysis

### Complete Coordinate Flow: ERA5 to CERRA

#### Stage 1: Data Loading
**File**: `src/weathergen/datasets/data_reader_anemoi.py`

```python
# Lines 101-103: Coordinates loaded from dataset
self.latitudes = _clip_lat(ds.latitudes)    # Native grid resolution
self.longitudes = _clip_lon(ds.longitudes)  # Native grid resolution
```

For CERRA: Loads full 5.5km resolution coordinates (~244,000 points per timestep over Europe)

#### Stage 2: Data Collection
**File**: `src/weathergen/datasets/multi_stream_data_sampler.py`

```python
# Line 426-427: Collect target data (CERRA)
rdata: IOReaderData = collect_datasources(stream_ds, step_forecast_dt, "target")
# rdata.coords contains FULL CERRA resolution at this point
```

#### Stage 3: Tokenization (CRITICAL)
**File**: `src/weathergen/datasets/tokenizer_masking.py`

```python
# Lines 126-136: Tokenize targets
tokenize_window = partial(
    tokenize_window_spacetime if tokenize_spacetime else tokenize_window_space,
    time_win=time_win,
    token_size=token_size,
    hl=self.hl_source,  # BUG: Uses SOURCE level for TARGET tokenization
    hpy_verts_rots=self.hpy_verts_rots_source[-1],
    ...
)
```

#### Stage 4: StreamData Storage
**File**: `src/weathergen/datasets/stream_data.py`

```python
# Line 205: Store raw coordinates
self.target_coords_raw[fstep] = torch.cat(target_coords_raw)
```

#### Stage 5: Model Prediction
**File**: `src/weathergen/model/model.py`

```python
# Lines 768-863: Predict at target coordinates
def predict(self, model_params, fstep, tokens, streams_data, target_coords_idxs):
    # embed target coordinates
    tc_tokens = checkpoint(tc_embed, streams_data[i_b][ii].target_coords[fstep], ...)

    # cross-attention: query latent with target coordinates
    tc_tokens = tte(latent=tokens_stream, output=tc_tokens, ...)

    # prediction head
    preds_tokens += [checkpoint(self.pred_heads[ii], tc_tokens, ...)]
```

#### Stage 6: Output Writing
**File**: `src/weathergen/utils/validation_io.py`

```python
# Lines 68-82: Write output with coordinates
data = io.OutputBatchData(
    sources,
    source_intervals,
    targets_all,
    preds_all,
    targets_coords_all,  # Coordinates from tokenization
    ...
)
```

---

## Current Resolution Issue

### The Problem

The model uses a single `healpix_level: 5` for both ERA5 input and CERRA output:

**Config**: `config/config_downscaling_era5_to_cerra.yml`
```yaml
healpix_level: 5
```

**Tokenizer**: `src/weathergen/datasets/tokenizer.py`
```python
def __init__(self, healpix_level: int):
    self.hl_source = healpix_level
    self.hl_target = healpix_level  # Both use same level
```

### Why This Limits Resolution

At HEALPix level 5:
- **Number of cells**: 12,288 globally
- **Cell size**: ~370km x 370km
- **ERA5 points per cell**: ~3-4 (good match)
- **CERRA points per cell**: ~4,500 (WAY too many!)

**Consequence**: The model creates ONE latent vector per cell. All ~4,500 CERRA points in a cell share this same latent. When the PerceiverIO decoder queries at different CERRA coordinates within the same cell, it sees nearly identical key/value pairs, producing nearly identical outputs.

### Visual Analogy

Imagine painting a detailed picture (CERRA resolution), but you can only choose ONE color per large tile (HEALPix cell). The canvas has fine detail, but the colors are uniform within each tile.

### Key Bug Location

**File**: `src/weathergen/datasets/tokenizer_masking.py` line 130
```python
hl=self.hl_source,  # Uses SOURCE level for TARGET tokenization
```

This should use `self.hl_target` to allow targets to be tokenized at a finer resolution.

---

## Implementation Fix

### Approach: Separate HEALPix Levels for Source and Target

Allow the encoder (source) to operate at coarse resolution while the decoder (target) operates at fine resolution.

### Important Caveats

> **Reality Check**: Even with separate target HEALPix level, the **latent grid still uses the coarse source level**. You should see more spatial variation within cells, but do not expect full 5.5km detail unless you also increase latent resolution or model capacity. The model learns to interpolate from coarse latent to fine target coordinates.

---

### Changes Required

#### 1. Config Change
**File**: `config/config_downscaling_era5_to_cerra.yml`
```yaml
healpix_level: 5           # For ERA5 source
healpix_level_target: 7    # For CERRA target (NEW)
```

**File**: `config/default_config.yml` (add for backward compatibility)
```yaml
healpix_level_target: null  # null means use healpix_level
```

#### 2. Tokenizer Base Class (CRITICAL - hpy_verts Bug)
**File**: `src/weathergen/datasets/tokenizer.py`

The current code **overwrites** `self.hpy_verts` when building target verts (line 63), losing the source version!

```python
def __init__(self, healpix_level: int, healpix_level_target: int | None = None):
    ref = torch.tensor([1.0, 0.0, 0.0])

    self.hl_source = healpix_level
    self.hl_target = healpix_level_target if healpix_level_target is not None else healpix_level

    self.num_healpix_cells_source = 12 * 4**self.hl_source
    self.num_healpix_cells_target = 12 * 4**self.hl_target

    # Build SOURCE verts
    verts00, verts00_rots = healpix_verts_rots(self.hl_source, 0.0, 0.0)
    # ... (other verts)
    vertsmm_source, vertsmm_rots = healpix_verts_rots(self.hl_source, 0.5, 0.5)

    self.hpy_verts_source = [...]  # NEW: Keep source verts separate!
    self.hpy_verts_rots_source = [...]

    # Build TARGET verts (only if different level)
    if self.hl_target != self.hl_source:
        verts00, verts00_rots = healpix_verts_rots(self.hl_target, 0.0, 0.0)
        # ... (other verts)
        vertsmm_target, vertsmm_rots = healpix_verts_rots(self.hl_target, 0.5, 0.5)
        self.hpy_verts_target = [...]  # NEW: Separate target verts
    else:
        self.hpy_verts_target = self.hpy_verts_source
        vertsmm_target = vertsmm_source

    self.hpy_verts_rots_target = [...]
```

**Also fix `compute_source_centroids`** (line 121) which currently uses `self.hpy_verts[-1]` (overwritten to target):
```python
def compute_source_centroids(self, source_tokens_cells: list[torch.Tensor]) -> torch.Tensor:
    source_means = [
        (
            self.hpy_verts_source[-1][i].unsqueeze(0).repeat(len(s), 1)  # FIXED: was hpy_verts
            if len(s) > 0
            else torch.tensor([])
        )
        for i, s in enumerate(source_tokens_cells)
    ]
    # ...
```

#### 3. StreamData Class (CRITICAL - Cell Count Mismatch)
**File**: `src/weathergen/datasets/stream_data.py`

The current code sizes target arrays using `healpix_cells` (source count). Need separate counts:

```python
def __init__(self, idx: int, forecast_steps: int, healpix_cells_source: int, healpix_cells_target: int) -> None:
    self.healpix_cells_source = healpix_cells_source
    self.healpix_cells_target = healpix_cells_target

    # Source arrays use source cell count
    self.source_tokens_lens = torch.zeros([healpix_cells_source], dtype=torch.int32)

    # Target arrays use TARGET cell count
    self.target_coords_lens = [
        torch.tensor([0 for _ in range(healpix_cells_target)]) for _ in range(forecast_steps + 1)
    ]
    self.target_tokens_lens = [
        torch.tensor([0 for _ in range(healpix_cells_target)]) for _ in range(forecast_steps + 1)
    ]
```

**Also update `add_empty_target`** (line 134) to use `healpix_cells_target`.

#### 4. MultiStreamDataSampler
**File**: `src/weathergen/datasets/multi_stream_data_sampler.py`

Update to pass both cell counts and use backward-compatible config:
```python
self.healpix_level: int = cf.healpix_level
self.healpix_level_target: int = cf.get('healpix_level_target', cf.healpix_level)  # Backward compatible
self.num_healpix_cells: int = 12 * 4**self.healpix_level
self.num_healpix_cells_target: int = 12 * 4**self.healpix_level_target

# Update StreamData instantiation (line ~387):
stream_data = StreamData(
    idx,
    forecast_dt + self.forecast_offset,
    self.num_healpix_cells,        # Source cells
    self.num_healpix_cells_target  # Target cells
)

# Update tokenizer instantiation:
if cf.training_mode == "masking":
    masker = Masker(cf)
    self.tokenizer = TokenizerMasking(cf.healpix_level, masker, self.healpix_level_target)

# Update target spoof (line ~437) to use target level:
rdata = spoof(
    self.healpix_level_target,  # FIXED: was healpix_level
    time_win_target.start,
    ...
)
```

#### 5. TokenizerMasking (Bug Fix)
**File**: `src/weathergen/datasets/tokenizer_masking.py`
```python
# Update __init__ to pass healpix_level_target to parent
def __init__(self, healpix_level: int, masker: Masker, healpix_level_target: int | None = None):
    super().__init__(healpix_level, healpix_level_target)

# Fix batchify_target (line 130)
hl=self.hl_target,  # FIXED: was self.hl_source
hpy_verts_rots=self.hpy_verts_rots_target[-1],  # FIXED: was _source
```

#### 6. TokenizerForecast
**File**: `src/weathergen/datasets/tokenizer_forecast.py`
```python
def __init__(self, healpix_level: int, healpix_level_target: int | None = None):
    super().__init__(healpix_level, healpix_level_target)
```

---

### Summary of All Changes

| File | Issue | Fix |
|------|-------|-----|
| `tokenizer.py` | `hpy_verts` overwritten, losing source | Keep separate `hpy_verts_source` and `hpy_verts_target` |
| `tokenizer.py` | `compute_source_centroids` uses wrong verts | Use `hpy_verts_source[-1]` |
| `stream_data.py` | Target arrays sized with source cells | Pass both cell counts, size target arrays with target count |
| `multi_stream_data_sampler.py` | Single healpix level | Read both levels, pass both to StreamData and tokenizers |
| `tokenizer_masking.py` | `batchify_target` uses source level | Use `hl_target` and `hpy_verts_rots_target` |
| `config` | No target level option | Add `healpix_level_target` with backward-compatible default |

### Why Model Changes Are Minimal

The PerceiverIO decoder uses cross-attention:
- **Query**: Target coordinate embeddings (arbitrary number, sized by target cells)
- **Key/Value**: Latent tokens from encoder (fixed by source healpix level)

This naturally supports different numbers of queries vs key/value pairs. The decoder learns to interpolate from coarse latent to fine target coordinates.

---

## Key Files Reference

### Core Tokenization
| File | Purpose |
|------|---------|
| `src/weathergen/datasets/tokenizer.py` | Base tokenizer class, HEALPix setup |
| `src/weathergen/datasets/tokenizer_masking.py` | Tokenizer for masking training mode |
| `src/weathergen/datasets/tokenizer_forecast.py` | Tokenizer for forecast training mode |
| `src/weathergen/datasets/tokenizer_utils.py` | HEALPix cell assignment, coordinate encoding |

### Data Pipeline
| File | Purpose |
|------|---------|
| `src/weathergen/datasets/data_reader_anemoi.py` | Loads data from Anemoi datasets |
| `src/weathergen/datasets/multi_stream_data_sampler.py` | Creates batches, orchestrates tokenization |
| `src/weathergen/datasets/stream_data.py` | Stores tokenized data per stream |

### Model
| File | Purpose |
|------|---------|
| `src/weathergen/model/model.py` | Main model class, prediction logic |
| `src/weathergen/model/engines.py` | Embedding, local/global attention, decoder |

### Output
| File | Purpose |
|------|---------|
| `src/weathergen/utils/validation_io.py` | Validation output writing |
| `packages/common/src/weathergen/common/io.py` | Zarr output utilities |

### Config
| File | Purpose |
|------|---------|
| `config/config_downscaling_era5_to_cerra.yml` | Main downscaling config |
| `config/streams/downscaling_era5_to_cerra/era5.yml` | ERA5 stream config |
| `config/streams/downscaling_era5_to_cerra/cerra.yml` | CERRA stream config |

---

## Memory Considerations

### HEALPix Level vs Memory (40GB A100)

| Target Level | Global Cells | CERRA Tokens/Sample* | Memory Est. | Recommendation |
|-------------|--------------|---------------------|-------------|----------------|
| 5 (current) | 12,288 | ~1,000 | ~2-3GB | Too coarse |
| 6 | 49,152 | ~5,000 | ~4-6GB | Still coarse |
| 7 | 196,608 | ~20,000 | ~8-12GB | Recommended |
| 8 | 786,432 | ~80,000 | ~20-30GB | May work |
| 9 | 3,145,728 | ~300,000 | OOM | Too fine |

*CERRA covers ~10% of global cells (Europe only)

### Memory Optimization Options

If level 7 or 8 causes OOM:

```yaml
# Reduce model dimensions
ae_local_dim_embed: 64   # from 128
ae_global_dim_embed: 128 # from 256

# Reduce target sampling
sampling_rate_target: 0.5  # from 1.0

# Reduce token size
# In cerra.yml:
token_size: 256  # from 512
```

### Gradient Checkpointing

The model already uses gradient checkpointing:
```python
# model.py
preds_tokens += [checkpoint(self.pred_heads[ii], tc_tokens, ...)]
```

---

## Stream Configuration

### ERA5 Stream (Source)
**File**: `config/streams/downscaling_era5_to_cerra/era5.yml`
```yaml
ERA5:
  type: anemoi
  filenames: ['aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr']
  source: ['2t', 'msl', '10u', '10v', ...]  # Input variables
  target: []  # No targets from ERA5
  token_size: 8  # Small: ERA5 is coarse
```

### CERRA Stream (Target)
**File**: `config/streams/downscaling_era5_to_cerra/cerra.yml`
```yaml
CERRA:
  type: anemoi
  filenames: ['cerra-rr-an-oper-se-al-ec-mars-5p5km-1985-2023-3h-v2.zarr']
  source: []  # No source from CERRA
  target: ['2t', 'msl', '10si', ...]  # Output variables
  diagnostic: True  # Marks as output-only stream
  token_size: 512  # Large: CERRA is fine resolution
```

---

## Verification After Fix

### 1. Training Check
```bash
python -m weathergen.train.train config/config_downscaling_era5_to_cerra.yml
# Monitor for OOM, check loss decreases
```

### 2. Output Inspection
```python
import zarr
import numpy as np

z = zarr.open("results/.../output.zarr")

# Get predictions and coordinates
preds = z['CERRA/target/data'][:]
coords = z['CERRA/target/coords'][:]

# Check that nearby coordinates have different predictions
# (Before fix: would be nearly identical within HEALPix cell)
print("Coordinate range:", coords.min(axis=0), coords.max(axis=0))
print("Prediction variance:", preds.var())

# Spatial variability check
from scipy.spatial import distance
nearby_mask = distance.cdist([coords[0]], coords)[0] < 50  # 50km
print("Variance in nearby predictions:", preds[nearby_mask].var())
```

### 3. Visual Comparison
Compare prediction maps to CERRA targets - should see spatial structure matching CERRA, not smooth ERA5-like patterns.

---

## Summary

### Root Cause
Single `healpix_level: 5` creates cells (~370km) much larger than CERRA resolution (5.5km). All CERRA points in a cell share one latent, so predictions lack spatial detail.

### Solution
Add `healpix_level_target` config option and fix the bug in `tokenizer_masking.py` where targets incorrectly use `hl_source`.

### Key Insight
The PerceiverIO decoder is resolution-agnostic - it can decode to arbitrary target coordinates. The limitation was in the tokenization, not the model architecture.
