# RADKLIM Dataset Configuration

## About RADKLIM
RADKLIM (Radar-Online-Anechoic-Process) provides hourly precipitation estimates over Germany based on weather radar data from DWD.

- **Spatial resolution**: ~1 km
- **Temporal resolution**: 1 hour
- **Coverage**: Germany
- **Time range**: 2001-2023
- **Data format**: Monthly netCDF files

## Dataset Structure

RADKLIM data is organized as:
```
radklim_base_path/
  2001/
    RW_2017.002_200101.nc
    RW_2017.002_200102.nc
    ...
  2002/
    ...
```

Each monthly file contains hourly precipitation for that month.

## Configuration Requirements

### 1. Base Path
Set the `data_path_radklim` in your private config to the base directory containing year folders.

Example:
```yaml
# In private config
DATA_PATH_RADKLIM: /p/data1/slmet/met_data/dwd/radklim-rw/netcdf/orig_grid
```

### 2. File Index (Recommended)
For faster loading, create an index file:

```bash
python build_radklim_index.py --input /path/to/radklim --output config/streams/radklim/radklim_file_index.json
```

This avoids scanning all files at startup.

## Memory Considerations

RADKLIM has very high spatial resolution:
- Native grid: ~990K points per timestep
- 6-hour window: 6 × 990K = ~6M points

### Recommended Settings
```yaml
spatial_stride: 2  # Reduces to ~247K points
max_num_targets: 100000  # Cap targets for memory
```

## Common Issues

### 1. Time alignment
RADKLIM timestamps may be offset (e.g., 00:50 instead of 00:00). Ensure your training window aligns with dataset times.

### 2. Missing data
Some files may be corrupted or missing. The reader will skip files with errors.

### 3. Memory
Always use spatial_stride and consider max_num_targets for training.

## Example Configuration

```yaml
RADKLIM:
  type: radklim
  filenames: ['']
  spatial_stride: 2
  index_file: config/streams/radklim/radklim_file_index.json
  token_size: 1024
```
