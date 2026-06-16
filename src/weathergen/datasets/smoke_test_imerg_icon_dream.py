"""
Smoke test: load a few samples from DataReaderImerg and DataReaderIconDream.

Run with:
    uv run python src/weathergen/datasets/smoke_test_imerg_icon_dream.py
"""

import numpy as np

from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.data_reader_icon_dream import DataReaderIconDream
from weathergen.datasets.data_reader_imerg import DataReaderImerg

DATA_ROOT = "/e/data1/slmet/ml_training"

# Short window inside both datasets' coverage (ICON-DREAM starts 2010, IMERG 1998).
T_START = np.datetime64("2015-06-01T00:00", "ns")
T_END = np.datetime64("2015-06-08T00:00", "ns")
WINDOW_LEN = np.timedelta64(1, "h").astype("timedelta64[ns]")
WINDOW_STEP = np.timedelta64(6, "h").astype("timedelta64[ns]")

N_SAMPLES = 3


def make_tw() -> TimeWindowHandler:
    return TimeWindowHandler(T_START, T_END, WINDOW_LEN, WINDOW_STEP)


def test_imerg() -> None:
    print("\n=== DataReaderImerg ===")
    stream_info = {
        "name": "IMERG_test",
        "source": ["precipitation"],
        "source_exclude": [],
        "target": [],
        "target_exclude": [],
        "spatial_bbox": [45.0, 55.0, 5.0, 15.0],  # small Germany box
        "spatial_stride": 2,
    }
    filename = f"{DATA_ROOT}/raina-imerg-nasa-0p1-1998-2025-30m-v1.zarr"
    reader = DataReaderImerg(make_tw(), filename, stream_info)

    print(f"  source channels : {reader.source_channels}")
    print(f"  grid points     : {reader.n_grid_points}")
    print(f"  data range      : {reader.data_start_time} → {reader.data_end_time}")

    idx_range = reader.time_window_handler.get_index_range()
    for i in range(min(N_SAMPLES, int(idx_range.end - idx_range.start))):
        rdata = reader.get_source(np.int64(i))
        print(
            f"  sample {i}: shape={rdata.data.shape}  "
            f"t={rdata.datetimes[0] if len(rdata.datetimes) else 'empty'}  "
            f"min={rdata.data.min():.4f}  max={rdata.data.max():.4f}"
        )

    print("  IMERG OK")


def test_icon_dream() -> None:
    print("\n=== DataReaderIconDream ===")
    stream_info = {
        "name": "ICON_DREAM_test",
        "source": ["10u", "10v", "2t", "tp"],
        "source_exclude": [],
        "target": [],
        "target_exclude": [],
        "spatial_bbox": [47.0, 55.0, 5.0, 15.0],  # Germany box
        "spatial_stride": 4,
        "geoinfo_channels": [],
        "parallel_reads": 2,
        "offset_cache_size": 64,
    }
    base_path = f"{DATA_ROOT}/ICON-DREAM"
    reader = DataReaderIconDream(make_tw(), base_path, stream_info)

    print(f"  source channels : {reader.source_channels}")
    print(f"  grid points     : {reader.n_grid_points}")
    print(f"  data range      : {reader.data_start_time} → {reader.data_end_time}")

    idx_range = reader.time_window_handler.get_index_range()
    for i in range(min(N_SAMPLES, int(idx_range.end - idx_range.start))):
        rdata = reader.get_source(np.int64(i))
        print(
            f"  sample {i}: shape={rdata.data.shape}  "
            f"t={rdata.datetimes[0] if len(rdata.datetimes) else 'empty'}  "
            f"min={rdata.data.min():.4f}  max={rdata.data.max():.4f}"
        )

    print("  ICON Dream OK")


if __name__ == "__main__":
    test_imerg()
    test_icon_dream()
    print("\nAll smoke tests passed.")
