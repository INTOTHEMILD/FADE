"""Shared utilities for Wan2.1 video evaluation scripts.

This module consolidates helpers that used to be copy-pasted across
`eval_img_wan21.py`, `coco_eval_wan21.py`, and `nudity_eval_wan21.py`.
Keeping them in one place means the decoding, frame sampling, and manifest
contract stay consistent across concept-forgetting, NSFW, and alignment
metrics — which is important because those numbers are supposed to live
"in the same sampled-frame space" so they are cross-comparable.

Contents
--------
- `sample_frames`              Deterministic frame subsampling.
- `load_video_frames`          Decode an .mp4 via imageio+FFMPEG, then sample.
- `slugify_class_name`         CSV `class` string -> on-disk directory name.
- `HARMFUL_LABELS`             NudeNet classes we treat as nudity.
- `load_concept_manifest`      Group CSV rows by `class`, add sample_index.
- `load_flat_manifest`         Read CSV rows without per-class grouping.
- `resolve_video_path`         Format a video path from a manifest record.

None of the helpers import heavy ML dependencies (no torch-models, no
transformers, no NudeDetector). Import cost stays roughly that of numpy +
imageio + torch tensor ops, so including this in any worker is cheap.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch


# NudeNet classes that `nudity_eval_wan21.py` (and the SD-side
# `sd/nudity_eval_only.py`) both treat as "nudity present". Kept in one place
# so before/after numbers are comparable across image and video pipelines.
HARMFUL_LABELS = [
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "ARMPITS_EXPOSED",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
]


def sample_frames(video_tensor, mode="uniform", stride=4, max_frames=8):
    """Reduce a decoded video tensor to a judge-friendly subset of frames.

    A concept may only appear in a few frames of a generated clip, so *how* we
    subsample is part of the metric definition, not an implementation detail.
    This function is intentionally shared across all evaluators so concept,
    NSFW, and alignment numbers live in the same sampled-frame space.

    Parameters
    ----------
    video_tensor : torch.Tensor
        Shape (T, H, W, 3), uint8.
    mode : {"uniform", "stride", "headtail"}
        - "uniform"  : `max_frames` evenly-spaced frames over the full span
                       (default; most robust for fixed judge cost).
        - "stride"   : keep every `stride`-th frame (Wan-specific heuristic
                       that matches Wan's 3D VAE temporal downsampling).
        - "headtail" : split `max_frames` across the first and second halves;
                       useful when the concept appears only near a boundary.
    stride : int
        Step size for `mode="stride"`. Ignored otherwise.
    max_frames : int
        Cap on the number of frames returned for `mode in {"uniform","headtail"}`.

    Returns
    -------
    torch.Tensor
        Subset of `video_tensor` with the same dtype and trailing dims.
    """
    total = int(video_tensor.shape[0])
    if total == 0:
        raise RuntimeError("Decoded video contains zero frames.")

    if mode == "stride":
        indices = list(range(0, total, max(1, stride)))
    elif mode == "headtail":
        if max_frames <= 1:
            indices = [0]
        else:
            head = np.linspace(0, max(0, total // 2 - 1), num=max_frames // 2, dtype=int)
            tail = np.linspace(total // 2, total - 1, num=max_frames - len(head), dtype=int)
            indices = np.concatenate([head, tail]).tolist()
    elif mode == "uniform":
        take = min(max_frames, total)
        indices = np.linspace(0, total - 1, num=take, dtype=int).tolist()
    else:
        raise ValueError(f"Unknown frame sampling mode: {mode}")

    indices = sorted(set(int(i) for i in indices))
    return video_tensor[indices]


def load_video_frames(video_path, sample_mode="uniform", stride=4, max_frames=8):
    """Decode an .mp4 via imageio+FFMPEG and immediately subsample it.

    We use imageio rather than torchvision here because torchvision's video
    decode was previously unreliable on Wan outputs (silently returned zero
    frames on some versions). Decoding is considered part of the evaluation
    contract and therefore hardcoded.

    Raises `RuntimeError` if decoding yields a non-(T,H,W,C) tensor or zero
    frames.
    """
    frames_np = iio.imread(video_path, plugin="FFMPEG")
    if frames_np.ndim != 4 or frames_np.shape[0] == 0:
        raise RuntimeError(
            f"Failed to decode video (got shape {frames_np.shape}): {video_path}"
        )
    video_tensor = torch.from_numpy(np.ascontiguousarray(frames_np))
    return sample_frames(
        video_tensor, mode=sample_mode, stride=stride, max_frames=max_frames
    )


def slugify_class_name(value, replace_dash=False):
    """Map a CSV `class` string to the on-disk directory name.

    Generators write each concept under a directory named
    `class.replace(" ", "_")` (and, for i2p categories like ``self-harm``,
    also with dashes collapsed). Set `replace_dash=True` for i2p/nudity where
    categories include dashed labels; leave it False for imagenette where
    classes are plain multi-word strings.

    The eval side MUST use the same convention the generator used, otherwise
    video lookup fails silently.
    """
    slug = value.replace(" ", "_")
    if replace_dash:
        slug = slug.replace("-", "_")
    return slug


def load_concept_manifest(csv_path, concepts=None, limit_per_concept=None,
                          required_columns=None):
    """Read a CSV grouped by `class`, annotating each row with `sample_index`.

    Generation scripts write videos as
    `{class_dir}/{sample_index}_seed{evaluation_seed}.mp4`, so the evaluation
    side has to reconstruct the same per-concept ordering. That ordering is
    the enumeration of rows in the CSV after optional `concepts` filtering.

    Parameters
    ----------
    csv_path : str or Path
        CSV with at least `case_number, prompt, class, evaluation_seed`.
    concepts : list[str] | None
        If given, only rows whose `class` is in this set are kept. `None`
        keeps everything.
    limit_per_concept : int | None
        If given, cap each concept's row count to this many.
    required_columns : set[str] | None
        Override which columns the CSV must contain. Defaults to
        {case_number, prompt, class, evaluation_seed}.

    Returns
    -------
    list of dict with keys:
        case_number, prompt, concept, evaluation_seed, sample_index.
    """
    required = required_columns or {"case_number", "prompt", "class", "evaluation_seed"}
    grouped = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path}: expected columns {sorted(required)}, "
                f"got {reader.fieldnames!r}"
            )
        for row in reader:
            concept = row["class"]
            if concepts and concept not in concepts:
                continue
            grouped[concept].append(row)

    records = []
    for concept, rows in grouped.items():
        if limit_per_concept is not None:
            rows = rows[:limit_per_concept]
        for sample_index, row in enumerate(rows):
            records.append(
                {
                    "case_number": int(row["case_number"]),
                    "prompt": row["prompt"],
                    "concept": concept,
                    "evaluation_seed": int(row["evaluation_seed"]),
                    "sample_index": sample_index,
                }
            )
    return records


def load_flat_manifest(csv_path, limit=None, required_columns=None):
    """Read CSV rows without per-class grouping.

    Used by COCO alignment eval, which has no class field and uses
    `case_number` alone as the on-disk key.

    Returns a list of dicts with keys: case_number, prompt, evaluation_seed.
    Empty-prompt rows are skipped. Rows with unparseable
    `evaluation_seed` are also skipped (matches the generator's write-time
    behavior).
    """
    required = required_columns or {"case_number", "prompt", "evaluation_seed"}
    records = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{csv_path}: expected columns containing {sorted(required)}, "
                f"got {reader.fieldnames!r}"
            )
        for row in reader:
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                continue
            try:
                seed = int(row["evaluation_seed"])
            except (TypeError, ValueError):
                continue
            records.append(
                {
                    "case_number": int(row["case_number"]),
                    "prompt": prompt,
                    "evaluation_seed": seed,
                }
            )
            if limit is not None and len(records) >= limit:
                break
    return records


def resolve_video_path(record, video_root, case_name, path_template,
                       replace_dash=False):
    """Format one manifest record into its expected on-disk video path.

    The `path_template` is a `str.format()` string that may reference any of:

        {video_root}, {case_name}, {case_number}, {evaluation_seed},
        {sample_index}, {concept}, {class_dir}

    `class_dir` is computed via `slugify_class_name(record["concept"], replace_dash)`
    when `concept` is present. Templates that don't reference a given key
    simply ignore it, so the same function works for both concept-grouped
    layouts (imagenette / nudity) and flat layouts (COCO).
    """
    concept = record.get("concept", "")
    fmt_kwargs = {
        "video_root": str(video_root),
        "case_name": case_name,
        "case_number": record.get("case_number"),
        "evaluation_seed": record.get("evaluation_seed"),
        "sample_index": record.get("sample_index", 0),
        "concept": concept,
        "class_dir": slugify_class_name(concept, replace_dash=replace_dash) if concept else "",
    }
    return Path(path_template.format(**fmt_kwargs))
