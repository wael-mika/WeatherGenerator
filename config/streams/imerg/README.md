# IMERG Dataset Configuration

## About IMERG
IMERG (Integrated Multi-satellitE Retrievals for GPM) provides global precipitation estimates at 0.1° spatial resolution and 30-minute temporal resolution.

- **Spatial resolution**: 0.1° (~11 km)
- **Temporal resolution**: 30 minutes
- **Coverage**: Global
- **Time range**: 1998-present
- **Data volume**: ~11.3 TB for full dataset

## Memory Considerations

**Important**: IMERG is extremely large. Without spatial filtering, the full global grid contains:

- 3600 longitudes × 1800 latitudes = 6.48 million grid points per timestep
- At 30-minute resolution, this is ~48 timesteps per day
- For a 6-hour window: 12 timesteps × 6.48M = **~78 million data points**

This will exceed memory limits for most systems.

## Recommended Settings

### Europe Region (Recommended)
```yaml
spatial_bbox: [35.0, 70.0, -10.0, 40.0]
spatial_stride: 2
```

This reduces data by ~98.7%:
- 6.48M → 325K (bbox) → 81K (stride=2)

### Germany Region (Smaller)
```yaml
spatial_bbox: [47.0, 55.0, 5.0, 15.0]
spatial_stride: 2
```

### Full Globe (Not Recommended)
```yaml
spatial_bbox: null
spatial_stride: 4
```

**Warning**: Requires 100+ GB RAM even with stride.

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `spatial_bbox` | [lat_min, lat_max, lon_min, lon_max] | None (full globe) |
| `spatial_stride` | Subsample grid (1=all, 2=every 2nd point) | 1 |
| `token_size` | Token size for batching | 1024 |
| `max_num_targets` | Limit target points per batch | -1 (no limit) |

## Performance Tips

1. **Always use spatial_bbox** unless you have 100+ GB RAM
2. **Increase spatial_stride** to further reduce memory
3. **Use smaller token_size** for finer spatial resolution
4. **Set max_num_targets** to cap memory usage
5. **Reduce batch_size** (often 1 for large regions)

## Example: Training on Europe

```yaml
spatial_bbox: [35.0, 70.0, -10.0, 40.0]
spatial_stride: 2
token_size: 1024
max_num_targets: 50000
```

This configuration should fit in ~20-30 GB GPU memory depending on model size.
