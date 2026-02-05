# RADKLIM Stream Configuration

Configuration for RADKLIM (DWD radar-based precipitation climatology) dataset in netCDF format.

## Dataset Information

- **Type:** `radklim`
- **Variable:** `RR` - Hourly rainfall (kg m⁻² / mm)
  - **Note:** Use `RR` as the channel name in your config (this is the actual netCDF variable name)
  - NetCDF attribute: `long_name: hourly rainfall`, `standard_name: rainfall_amount`
- **Spatial Resolution:** ~1 km (RADOLAN grid)
- **Temporal Resolution:** 1 hour
- **Coverage:** Germany and surroundings (2001-2023)
- **Data Format:** netCDF (monthly files)
- **Grid Size:** 1100 × 900 points
- **Total Size:** ~20.82 GB

## Usage

### Basic Configuration

Use this stream in your main config by including:

```yaml
data_path_radklim: /p/data1/slmet/met_data/dwd/radklim-rw/netcdf/orig_grid

streams:
  - !include config/streams/radklim/radklim.yml
```

Or directly specify:

```yaml
streams:
  - name: RADKLIM
    type: radklim
    filenames: ['']  # Empty or '.' - base path is used directly
    source: ['RR']  # RR is the netCDF variable name for hourly rainfall
    target: ['RR']
    frequency: "01:00:00"
```

### Configuration Options

- **source:** List of variables to use as model input (default: `['RR']`)
- **target:** List of variables to use as prediction target (default: `['RR']`)
- **source_exclude:** Variables to exclude from source (default: `[]`)
- **target_exclude:** Variables to exclude from target (default: `[]`)
- **frequency:** Time resolution (default: `"01:00:00"` = 1 hour)

### File Structure

RADKLIM data is organized as:
```
/p/data1/slmet/met_data/dwd/radklim-rw/netcdf/orig_grid/
├── 2001/
│   ├── RW_2017.002_200101.nc
│   ├── RW_2017.002_200102.nc
│   └── ...
├── 2002/
│   └── ...
...
└── 2023/
    └── ...
```

The data reader automatically discovers and indexes all monthly files.

## Dataset Details

### Spatial Coverage

- **Region:** Germany and surrounding areas
- **Projection:** Polar stereographic (RADOLAN grid)
- **Coordinates:**
  - Latitude: 46.19° to 55.78° N
  - Longitude: 3.10° to 17.10° E

### Temporal Coverage

- **Period:** 2001-01-01 to 2023-12-31
- **Files:** 276 monthly netCDF files (23 years × 12 months)
- **Timesteps per file:** ~744 (31 days × 24 hours)
- **Time offset:** Data timestamped at :50 minutes past each hour

### Data Quality

- **Fill value:** 999.0 (automatically converted to NaN)
- **Missing data:** Some timesteps may have NaN values for certain regions

## Performance Characteristics

- **Grid size:** 990,000 points per timestep (moderate)
- **Loading time:** Fast - only ~2-5 seconds per 6-hour window
- **Memory usage:** ~150 MB per 6-hour window (6 timesteps)
- **Multi-file handling:** Efficient caching minimizes I/O

### Performance vs. IMERG

RADKLIM is **significantly faster** than IMERG because:
- **Smaller grid:** 990K vs. 6.48M points (~7× smaller)
- **Regional coverage:** Germany only vs. global
- **File caching:** Only one file typically open at a time

## Examples

### Source Only (Input Features)
```yaml
RADKLIM:
  type: radklim
  filenames: ['']
  source: ['RR']
  target_exclude: ['RR']  # Don't use as target
```

### Target Only (Prediction)
```yaml
RADKLIM:
  type: radklim
  filenames: ['']
  source_exclude: ['RR']  # Don't use as source
  target: ['RR']
```

### Custom Time Range
```yaml
# In main config
start_date: "2015-01-01"
end_date: "2020-12-31"

streams:
  - !include config/streams/radklim/radklim.yml
```

## Dataset Citation

```
Deutscher Wetterdienst (DWD)
RADKLIM - Radar-based precipitation climatology
Version 2017.002
Institution: Deutscher Wetterdienst
Reference: DOI 10.5676/DWD/RADKLIM_RW_V2017.002
Contact: Harald Rybka, Katharina Lengfeld
```

## Related Files

- **Data Reader:** `src/weathergen/datasets/data_reader_radklim.py`
- **Test Script:** `test_radklim_reader.py`
- **Design Document:** `RADKLIM_DATA_READER_PLAN.md`
- **Dataset Inspector:** `inspect_radklim_netcdf.py`

## Troubleshooting

### Issue: FileNotFoundError

**Cause:** Base path not found or incorrect

**Solution:** Verify `data_path_radklim` in config points to the correct directory
```yaml
data_path_radklim: /p/data1/slmet/met_data/dwd/radklim-rw/netcdf/orig_grid
```

### Issue: No data for time range

**Cause:** Requested time range outside available data (2001-2023)

**Solution:** Check time range in config:
```yaml
start_date: "2010-01-01"  # Must be >= 2001
end_date: "2020-12-31"    # Must be <= 2023
```

### Issue: Slow initialization

**Cause:** Building file index requires scanning all 276 files

**Solution:** This is normal on first load (~5-10 seconds). Consider:
- Caching file index to disk (future enhancement)
- Using smaller time range if full period not needed

### Issue: Many NaN values

**Expected behavior:** RADKLIM data has NaN values for:
- Areas outside radar coverage
- Invalid or quality-flagged measurements
- Some timesteps during maintenance

To filter NaN:
```python
source_data = reader.get_source(0).remove_nan_coords()
```

## Integration with WeatherGenerator

The RADKLIM reader is fully integrated and works with:

- ✅ MultiStreamDataSampler
- ✅ Tokenization (HEALPix grid)
- ✅ Masking strategies
- ✅ Forecasting workflows
- ✅ Multi-stream training
- ✅ Multi-file handling (seamless across month boundaries)

## Next Steps

After setting up RADKLIM in your configuration:

1. **Verify data access:** Ensure directory is accessible
   ```bash
   ls /p/data1/slmet/met_data/dwd/radklim-rw/netcdf/orig_grid/
   ```

2. **Run tests:** Validate the reader implementation
   ```bash
   python test_radklim_reader.py
   ```

3. **Train model:** Use RADKLIM like any other data stream

## Notes

- RADKLIM uses polar stereographic projection but provides lat/lon coordinates
- Data reader handles projection automatically
- Multi-month windows are supported seamlessly
- File caching minimizes repeated I/O operations
