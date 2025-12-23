# IMERG Stream Configuration

Configuration for IMERG (Integrated Multi-satellitE Retrievals for GPM) precipitation data.

## Dataset Information

- **Type:** `imerg`
- **Variable:** Precipitation (mm)
- **Spatial Resolution:** 0.1° × 0.1°
- **Temporal Resolution:** 30 minutes
- **Coverage:** Global (1998-2025)
- **Data Format:** Zarr
- **Size:** ~11.6 TB

## Usage

### Basic Configuration

Use this stream in your main config by including:

```yaml
data_path_imerg: /p/scratch/weatherai/shared/RAINA

streams:
  - !include config/streams/imerg/imerg.yml
```

Or directly specify:

```yaml
streams:
  - name: IMERG
    type: imerg
    filenames: ['raina-imerg-nasa-0p1-1998-2025-30m-v1.zarr']
    source: ['precipitation']
    target: ['precipitation']
    frequency: "00:30:00"
```

### Configuration Options

- **source:** List of variables to use as model input (default: `['precipitation']`)
- **target:** List of variables to use as prediction target (default: `['precipitation']`)
- **source_exclude:** Variables to exclude from source (default: `[]`)
- **target_exclude:** Variables to exclude from target (default: `[]`)
- **frequency:** Time resolution (default: `"00:30:00"` = 30 minutes)

### Performance Tips

⚠️ **Note:** IMERG has a very large spatial grid (6.48M points per timestep). Loading can be slow.

**Recommended for faster loading:**
```yaml
# Use smaller time windows
len_hrs: 1  # Instead of 6

# Or larger time steps
step_hrs: 24  # Daily instead of 6-hourly
```

See `IMERG_USAGE_EXAMPLE.md` in the root directory for more details.

## Dataset Citation

```
Mohamad Hakam Shams Eddin, Anas Allahham (2025)
RAINA IMERG NASA 0.1° 1998-2025 30-minute v1
Contact: shams@iai.uni-bonn.de, allahham@iai.uni-bonn.de
Institution: Uni Bonn - Institute of Computer Science III
Data Source: NASA GPM IMERG (https://gpm.nasa.gov/data/imerg)
```

## Examples

### Source Only (Input Features)
```yaml
IMERG:
  type: imerg
  filenames: ['raina-imerg-nasa-0p1-1998-2025-30m-v1.zarr']
  source: ['precipitation']
  target_exclude: ['precipitation']  # Don't use as target
```

### Target Only (Prediction)
```yaml
IMERG:
  type: imerg
  filenames: ['raina-imerg-nasa-0p1-1998-2025-30m-v1.zarr']
  source_exclude: ['precipitation']  # Don't use as source
  target: ['precipitation']
```

## Related Files

- **Data Reader:** `src/weathergen/datasets/data_reader_imerg.py`
- **Test Script:** `test_imerg_reader.py`
- **Usage Guide:** `IMERG_USAGE_EXAMPLE.md`
- **Implementation Plan:** `IMERG_DATA_READER_PLAN.md`
