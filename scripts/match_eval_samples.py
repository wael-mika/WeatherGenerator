#!/usr/bin/env python3
# ruff: noqa: T201
"""Date-match samples across WeatherGenerator inference runs.

Inference runs written with different world sizes / num_workers emit their samples in a
different order, so sample index i is NOT the same date in two runs. This reads the zarr
outputs directly, builds the (global sample index -> valid date) map for each run the same
way `WeatherGenZarrReader` does (rank files sorted, contiguous global offsets), and prints
the common dates plus per-run `sample:` lists ready to paste into an eval config.

A sample only counts if EVERY requested forecast step has both prediction and target data:
a time-limited inference leaves a trailing sample with step 1 written but not steps 2-3, and
the eval reader's presence check does not catch it (it fails later with a raw zarr KeyError).

  scripts/match_eval_samples.py azsw18eo le3u2yik
  scripts/match_eval_samples.py azsw18eo le3u2yik jepajune --stream ERA5 --fsteps 1-3
"""

import argparse
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import zarr

RESULTS = Path(
    os.environ.get("WG_RESULTS_DIR", "/iopsstor/scratch/cscs/thunter/shared_work/results")
)


def _rank_files(run_dir: Path, mini_epoch: int) -> list[Path]:
    """Rank files in the same order the evaluate reader discovers them."""
    files = sorted(run_dir.glob(f"validation_chkpt{mini_epoch:05d}_rank*.zip"))
    files += sorted(run_dir.glob(f"validation_chkpt{mini_epoch:05d}_rank*.zarr"))
    if not files:
        raise FileNotFoundError(f"no rank files for {run_dir}")
    return files


def _is_complete(sample_group, stream: str, fsteps: list[int]) -> bool:
    """True when every requested forecast step has prediction AND target data."""
    for fstep in fsteps:
        try:
            step = sample_group[stream][str(fstep)]
            if 0 in (step["prediction"]["data"].shape[0], step["target"]["data"].shape[0]):
                return False
        except (KeyError, FileNotFoundError):
            return False
    return True


def sample_dates(
    run_id: str, stream: str, fsteps: list[int], mini_epoch: int
) -> tuple[dict[int, str], list[int]]:
    """Map global sample index -> validity date of the first fstep, plus dropped indices.

    Global indices count every sample the reader would see, complete or not, so they stay
    consistent with the indices an eval config addresses; incomplete ones are reported
    separately rather than silently renumbering the rest.
    """
    out: dict[int, str] = {}
    dropped: list[int] = []
    offset = 0
    for f in _rank_files(RESULTS / run_id, mini_epoch):
        store = zarr.storage.ZipStore(str(f), mode="r") if f.suffix == ".zip" else str(f)
        group = zarr.open_group(store=store, mode="r")
        local = sorted(int(k) for k in group.keys())
        for i, smp in enumerate(local):
            sample_group = group[str(smp)]
            if not _is_complete(sample_group, stream, fsteps):
                dropped.append(offset + i)
                continue
            times = sample_group[stream][str(fsteps[0])]["target"]["times"]
            out[offset + i] = str(np.datetime64(np.array(times[0]), "s"))
        offset += len(local)
    return out, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs="+", help="inference run ids to match")
    parser.add_argument("--stream", default="ERA5")
    parser.add_argument(
        "--fsteps",
        default="1-3",
        help="forecast steps that must all be present, e.g. '1-3' or '1'; "
        "the first one supplies the date key",
    )
    parser.add_argument("--mini-epoch", type=int, default=0)
    args = parser.parse_args()

    if "-" in args.fsteps:
        first, last = (int(v) for v in args.fsteps.split("-"))
        fsteps = list(range(first, last + 1))
    else:
        fsteps = [int(args.fsteps)]

    results = OrderedDict(
        (r, sample_dates(r, args.stream, fsteps, args.mini_epoch)) for r in args.run_ids
    )
    maps = OrderedDict((r, m) for r, (m, _) in results.items())

    print(f"complete = every fstep in {fsteps} has prediction and target data\n")
    for run_id, (m, dropped) in results.items():
        dates = sorted(m.values())
        drop_note = f", dropped incomplete {dropped}" if dropped else ""
        print(f"{run_id}: {len(m)} complete samples, {dates[0]} .. {dates[-1]}{drop_note}")

    common = set.intersection(*(set(m.values()) for m in maps.values()))
    print(f"\ncommon dates: {len(common)}")
    if not common:
        print("NO OVERLAP — the runs cover disjoint periods; re-run inference to match.")
        return

    for date in sorted(common)[:10]:
        idxs = ", ".join(
            f"{r}={next(i for i, d in m.items() if d == date)}" for r, m in maps.items()
        )
        print(f"  {date}  {idxs}")
    if len(common) > 10:
        print(f"  ... {len(common) - 10} more")

    print("\nper-run eval `sample:` lists (same order = same dates):")
    lists = {}
    for run_id, m in maps.items():
        inv = {d: i for i, d in m.items()}
        lists[run_id] = [inv[d] for d in sorted(common)]
        print(f"  {run_id}: {lists[run_id]}")

    # The convenient case: every run puts the same date at the same index, so one shared
    # `sample:` entry works. Order inside that entry is irrelevant (the reader sorts), so
    # judge by the index SET, not the listed order.
    aligned = len({frozenset(v) for v in lists.values()}) == 1
    indices = sorted(next(iter(lists.values())))
    contiguous = aligned and indices == list(range(len(indices)))
    if contiguous:
        print(f'\nVERDICT: index-aligned and contiguous -> sample: "0-{len(indices) - 1}"')
    elif aligned:
        print(f"\nVERDICT: index-aligned but not contiguous -> sample: {indices}")
    else:
        print("\nVERDICT: NOT index-aligned -> per-run sample lists above are required")


if __name__ == "__main__":
    main()
