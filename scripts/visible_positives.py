"""Scan a retained corpus for registered and still-uncovered visible marks.

Why this exists separately from `visible_removal_audit.py`: that audit is single-process,
so repeated full-dataset sweeps waste time. Its
expensive half is DETECTION, and detection does not depend on the fill backend. Splitting
it out means detecting once in parallel and then feeding the positives to the audit via
its `--paths-file` seam, over a few thousand images instead of forty thousand.

CRASH TOLERANCE IS NOT OPTIONAL AT THIS SCALE
  cv2/libpng can crash natively on malformed images. A plain `ProcessPoolExecutor.map`
  over a large dataset can deadlock: the worker dies without a Python traceback and the parent
  waits forever on a result that never arrives (observed 2026-07-19 -- 26 min of work lost
  because results were only written at the end). So this script:
    * writes every result to JSONL as it arrives -- a kill never costs more than a batch;
    * is resumable -- an interrupted run skips what it already recorded;
    * runs each image in a fresh interpreter with a per-image timeout and bounded
      concurrency; a poisoned file is recorded without retrying it in the parent.

Treat input datasets as sensitive and read-only, and keep output gitignored.

The JSONL is also the input to visible-mark discovery. Each row records strict and
metadata-corroborated detector results plus the metadata cohort independently of
the pixels. ``--sheets`` then makes blinded, native-detail top/bottom bands from:

* quiet TC260 producer cohorts, including producer identities absent from the
  visible-mark registry;
* quiet named-platform cohorts;
* an unbiased sample of every other quiet image, so a completely novel mark with
  no useful metadata is not excluded by construction.

The sheets are candidates for human adjudication, not evidence that a watermark is
present. Their manifest is deliberately separate and must stay closed until the
image review finishes.

    uv run python scripts/visible_positives.py --corpus .local-eval/originals --jobs 6
    uv run python scripts/visible_positives.py --since 2026-09-01 --sheets .local-eval/visible-fresh-sheets
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from _corpus_scan import corpus_files, repair_jsonl_tail
from _isolated_image_workers import run_batch

# The package's own format set. An inlined copy here silently skipped .heif, which
# CLAUDE.md documents as supported.
from remove_ai_watermarks._internal.constants import SUPPORTED_FORMATS as _EXTS

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / ".local-eval" / "originals"
OUT = REPO / ".local-eval" / "visible-positives.jsonl"
_SHEET_BAND_FRACTION = 0.12
_SCHEMA_VERSION = 2


def _one(path: str) -> dict[str, object]:
    from remove_ai_watermarks.api import _provenance_from_report
    from remove_ai_watermarks.identify import identify
    from remove_ai_watermarks.image_io import imread
    from remove_ai_watermarks.metadata import aigc_label, uscc_of
    from remove_ai_watermarks.watermark_registry import _provenance_confirms_product, known_marks

    try:
        source = Path(path)
        img = imread(source)
        if img is None:
            return _error_record(path, "unreadable")
        report = identify(source, check_visible=False, check_invisible=False)
        provenance = _provenance_from_report(report, source)
        strict = []
        accepted = []
        for mark in known_marks():
            strict_detection, relaxed_detection = mark.detect_both(img)
            if strict_detection.detected:
                strict.append(mark.key)
            detection = (
                relaxed_detection if _provenance_confirms_product(mark.product, provenance) else strict_detection
            )
            if detection.detected:
                accepted.append(mark.key)
        label = aigc_label(source) or {}
        producer = str(label.get("ContentProducer") or "")
        return {
            "schema_version": _SCHEMA_VERSION,
            "path": path,
            "keys": accepted,
            "strict_keys": strict,
            "provenance": sorted(provenance),
            "platform": report.platform or "",
            "producer": producer,
            "uscc": uscc_of(producer) if producer else "",
            "status": "ok",
            "error": "",
        }
    except Exception as e:
        return _error_record(path, f"error:{type(e).__name__}", str(e))


def _error_record(path: str, status: str, error: str = "") -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "path": path,
        "keys": [],
        "strict_keys": [],
        "provenance": [],
        "platform": "",
        "producer": "",
        "uscc": "",
        "status": status,
        "error": error[:300].replace("\n", " "),
    }


def _run_batch(batch: list[str], jobs: int, timeout: int) -> list[dict[str, object]]:
    """Bound each image in an independent interpreter, including native failures."""
    return run_batch(Path(__file__), batch, jobs=jobs, timeout=timeout)


def _files(corpus: Path, since: date | None) -> list[Path]:
    return corpus_files(corpus, since, suffixes=_EXTS)


def _read_records(path: Path) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(row, dict) and isinstance(row.get("path"), str):
                rows[row["path"]] = row
    return list(rows.values())


def _repair_checkpoint(path: Path) -> None:
    """Drop a torn tail before append, otherwise the next row joins invalid JSON."""
    repair_jsonl_tail(path)


def _discovery_candidates(
    rows: list[dict[str, Any]], *, per_cohort: int, random_quiet: int, seed: int
) -> list[dict[str, Any]]:
    """Stratify quiet rows without treating provenance as a pixel-level label."""
    rng = random.Random(seed)  # noqa: S311 -- reproducible review sampling
    quiet = [row for row in rows if row.get("status") == "ok" and not row.get("keys")]
    strata: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in quiet:
        if row.get("uscc"):
            strata[f"tc260:{row['uscc']}"].append(row)
        if row.get("platform"):
            strata[f"platform:{row['platform']}"].append(row)

    selected: dict[str, dict[str, Any]] = {}
    labels: dict[str, set[str]] = collections.defaultdict(set)
    for label, members in sorted(strata.items()):
        for row in rng.sample(members, k=min(per_cohort, len(members))):
            selected[row["path"]] = row
            labels[row["path"]].add(label)

    for row in rng.sample(quiet, k=min(random_quiet, len(quiet))):
        selected[row["path"]] = row
        labels[row["path"]].add("unbiased-quiet")

    return [{**row, "discovery_strata": sorted(labels[path])} for path, row in sorted(selected.items())]


def _bands(path: str) -> tuple[Any, Any] | None:
    from remove_ai_watermarks.image_io import imread

    image = imread(path)
    if image is None:
        return None
    height = image.shape[0]
    band_height = max(24, int(height * _SHEET_BAND_FRACTION))
    return image[:band_height], image[height - band_height :]


def _write_discovery_sheets(rows: list[dict[str, Any]], output: Path) -> int:
    import csv

    import cv2
    import numpy as np

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"discovery output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    written = 0
    for row in rows:
        bands = _bands(row["path"])
        if bands is None:
            continue
        top, bottom = bands
        width = max(top.shape[1], bottom.shape[1])
        divider = np.full((6, width, 3), 90, np.uint8)
        header = np.full((28, width, 3), 40, np.uint8)
        cv2.putText(header, f"{written:05d}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.imwrite(str(output / f"{written:05d}.png"), np.vstack((header, top, divider, bottom)))
        manifest.append({"idx": written, **row})
        written += 1
    if manifest:
        fields = list(manifest[0])
        with (output / "MANIFEST_DO_NOT_OPEN.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(manifest)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--since", type=date.fromisoformat, help="include dated corpus directories on or after YYYY-MM-DD")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--timeout", type=int, default=600, help="seconds per image before terminating its isolated worker")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--sheets", type=Path, help="write blinded discovery bands after the scan")
    ap.add_argument("--per-cohort", type=int, default=12, help="quiet images sampled from each metadata cohort")
    ap.add_argument("--random-quiet", type=int, default=400, help="unbiased quiet images sampled across the corpus")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    files = [str(path) for path in _files(a.corpus, a.since)]
    if a.limit:
        files = files[: a.limit]

    done: set[str] = set()
    if a.restart and a.out.exists():
        a.out.unlink()  # --restart must TRUNCATE; the file is reopened in append mode below
    _repair_checkpoint(a.out)
    if a.out.exists() and not a.restart:
        done = {row["path"] for row in _read_records(a.out) if row.get("schema_version") == _SCHEMA_VERSION}
    todo = [p for p in files if p not in done]
    print(f"images {len(files)}  already done {len(done)}  to do {len(todo)}  jobs {a.jobs}", flush=True)

    seen = 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as fh:
        for start in range(0, len(todo), a.batch):
            for rec in _run_batch(todo[start : start + a.batch], a.jobs, a.timeout):
                fh.write(json.dumps(rec) + "\n")
                seen += 1
            fh.flush()
            print(f"  {seen}/{len(todo)}", flush=True)

    records = _read_records(a.out)
    selected_paths = set(files)
    records = [row for row in records if row["path"] in selected_paths]
    hits = [row["path"] for row in records if row.get("keys")]
    counts = collections.Counter(str(key) for row in records for key in row.get("keys", []))
    strict_counts = collections.Counter(str(key) for row in records for key in row.get("strict_keys", []))
    status = collections.Counter(str(row.get("status")) for row in records)
    # Derive the paths file from --out so a trial run with a scratch --out cannot
    # overwrite the shared list a full sweep produced.
    paths_out = a.out.with_suffix(".txt")
    paths_tmp = paths_out.with_suffix(paths_out.suffix + ".tmp")
    paths_tmp.write_text("\n".join(sorted(set(hits))) + ("\n" if hits else ""), encoding="utf-8")
    paths_tmp.replace(paths_out)
    print(f"\npositives: {len(set(hits))} images")
    for k, v in counts.most_common():
        print(f"   {v:6d}  {k} ({strict_counts[k]} strict)")
    print(f"statuses: {dict(status)}\npaths -> {paths_out}\nrecords -> {a.out}")
    if a.sheets:
        candidates = _discovery_candidates(
            records,
            per_cohort=a.per_cohort,
            random_quiet=a.random_quiet,
            seed=a.seed,
        )
        written = _write_discovery_sheets(candidates, a.sheets)
        print(f"discovery candidates: {len(candidates)}  sheets written: {written} -> {a.sheets}")

    from remove_ai_watermarks.watermark_registry import mark_keys

    git = shutil.which("git")
    commit = (
        subprocess.run(  # noqa: S603 -- resolved git binary and fixed read-only arguments
            [git, "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if git
        else ""
    )
    metadata = {
        "command": [str(Path(__file__).relative_to(REPO)), *sys.argv[1:]],
        "library_commit": commit or None,
        "library_version": version("remove-ai-watermarks"),
        "corpus": str(a.corpus.resolve()),
        "since": a.since.isoformat() if a.since else None,
        "selected_images": len(files),
        "complete_records": len(records),
        "registry_keys": mark_keys(),
        "schema_version": _SCHEMA_VERSION,
    }
    a.out.with_suffix(a.out.suffix + ".run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
