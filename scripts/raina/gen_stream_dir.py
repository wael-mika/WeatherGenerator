#!/usr/bin/env python3
# ruff: noqa: T201
"""Generate a finetune stream dir VERBATIM from a pretrained run's saved config.

Writes config/streams/<out_dir>/<stream>.yml for every PRETRAINING INPUT (forcing) stream,
drops the pretraining OUTPUT stream(s), and copies imerg_anemoi.yml in as the sole
diagnostic/output stream. That is the "inputs + IMERG only" probe design.

Materializing the checkpoint's EXPLICIT channel lists (instead of replaying source_exclude
patterns) is deliberate: PR #2361 changed exclude matching from substring to exact, so
replaying old patterns silently leaks channels and breaks the embedder shapes -- that is
exactly how run nfwefvyx died ([25,512] checkpoint vs [22,512] model).

  scripts/raina/gen_stream_dir.py gkm6as6m imerg_diag_gkm6as6m
  scripts/raina/gen_stream_dir.py rck9wgm7 imerg_diag_rck9wgm7 --check   # regression check

--check regenerates into a temp dir and diffs against the existing dir instead of writing;
use it to confirm the generator still reproduces a known-good dir after editing this script.
"""

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("WG_MODELS_DIR", "/iopsstor/scratch/cscs/thunter/shared_work/models"))

# set at runtime by the reader/launcher, never authored by hand
DERIVED_KEYS = {
    "name",
    "data_paths",
    "train_source_channels",
    "train_target_channels",
    "val_source_channels",
    "val_target_channels",
}

HEADER = """# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
# {stream}: input-only (forcing) stream reproduced VERBATIM from the
# saved config of pretrained run {run_id}
# ({json_path}),
# minus runtime-derived keys. Do not edit by hand -- regenerate from the
# checkpoint config if the source run changes.

"""


def emit(run_id: str, out: Path, imerg_from: str) -> tuple[list[str], list[str]]:
    json_path = MODELS / run_id / f"model_{run_id}_latest.json"
    with open(json_path) as fh:
        cfg = json.load(fh)
    out.mkdir(parents=True, exist_ok=True)

    kept, dropped = [], []
    for name, s in cfg["streams"].items():
        if not s.get("forcing", False):
            dropped.append(name)  # the pretraining output stream; IMERG replaces it
            continue
        clean = {k: v for k, v in s.items() if k not in DERIVED_KEYS}
        body = yaml.dump({name: clean}, default_flow_style=None, sort_keys=False, width=100)
        (out / f"{name.lower()}.yml").write_text(
            HEADER.format(stream=name, run_id=run_id, json_path=json_path) + body
        )
        kept.append(name)

    shutil.copy2(REPO / imerg_from, out / Path(imerg_from).name)
    ids = {n: cfg["streams"][n].get("stream_id") for n in kept}
    assert 40 not in ids.values(), f"IMERG stream_id 40 collides with inherited ids {ids}"
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("out_dir", help="directory name under config/streams/")
    ap.add_argument("--imerg-from", default="config/streams/imerg_diag_rck9wgm7/imerg_anemoi.yml")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        tmp = Path(tempfile.mkdtemp())
        emit(args.run_id, tmp, args.imerg_from)
        existing = REPO / "config/streams" / args.out_dir
        names = sorted({p.name for p in tmp.iterdir()} | {p.name for p in existing.iterdir()})
        same = True
        for n in names:
            a, b = tmp / n, existing / n
            if not a.exists() or not b.exists():
                print(f"  ONLY IN {'generated' if a.exists() else 'existing'}: {n}")
                same = False
            elif filecmp.cmp(a, b, shallow=False):
                print(f"  identical: {n}")
            else:
                print(f"  DIFFERS  : {n}")
                subprocess.run(["diff", str(b), str(a)])
                same = False
        print("\nREPRODUCED EXACTLY" if same else "\ndifferences above (a zarr rename is OK)")
        return

    kept, dropped = emit(args.run_id, REPO / "config/streams" / args.out_dir, args.imerg_from)
    print(f"{args.run_id} -> config/streams/{args.out_dir}/")
    print(f"  kept (forcing inputs)  : {kept}")
    print(f"  DROPPED (output stream): {dropped}")
    print(f"  + {Path(args.imerg_from).name} (sole diagnostic output)")
    print("\nCHECK the zarr filenames in the generated ymls -- datasets get renamed/deleted.")


if __name__ == "__main__":
    main()
