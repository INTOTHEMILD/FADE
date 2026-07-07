#!/usr/bin/env python
"""Calibrate τ_c, T_c per concept and compute h_c_anchor.

Output: torch.save dict at <output>/routing_config.pt with keys:
    {"anchors": {c: tensor[d]}, "tau": {c: float}, "T": {c: float}, "diag": {...}}
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wan.configs import WAN_CONFIGS
from wan.modules.lora_router import (
    compute_concept_anchor,
    estimate_global_mean,
    pool_prompt_context,
)
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel


DEFAULT_TEMPLATES = [
    "<c>",
    "a photo of <c>",
    "an image of <c>",
    "a <c>",
]
SYNONYMS = {
    "english springer spaniel": ["a dog", "a puppy", "a spaniel"],
    "parachute": ["a skydiver", "a parachutist"],
    "church": ["a chapel", "a cathedral"],
    "Van Gogh": ["a painting", "an oil painting"],
    "anime": ["a cartoon", "anime drawing"],
    "Monet": ["impressionist painting", "an oil painting", "a painting"],
    "Picasso": ["cubist painting", "an abstract painting", "a painting"],
    "Da Vinci": ["Renaissance painting", "an oil painting", "a portrait"],
    "Salvador Dali": ["surrealist painting", "a dream painting", "a painting"],
    "naked person": ["a nude person", "a bare body", "an unclothed figure"],
}


def calibrate_one(
    concept, unsafe_csv, neutral_csv, text_encoder, text_embedding, device,
    global_mean=None, extra_negative_csvs=None,
):
    """Calibrate τ_c, T_c for one concept.

    extra_negative_csvs: list of unsafe CSV paths for *other* concepts in the
    same unlearning set. Including them as hard negatives is REQUIRED for
    multi-concept routing — otherwise τ is set only to separate the concept
    from generic motion prompts, and short anchors (e.g. "anime", "church"
    with 1-2 tokens) end up firing on every other concept's prompt as well.
    Without cross-concept negatives, parachute τ landed at 0.28 and fired
    at 0.9 on Ghibli anime prompts (see ablation_log § A1.8).
    """
    templates = DEFAULT_TEMPLATES + SYNONYMS.get(concept, [])
    h_anchor = compute_concept_anchor(
        concept, templates, text_encoder, text_embedding, device,
        global_mean=global_mean,
    )

    unsafe_prompts = pd.read_csv(unsafe_csv)["prompt"].tolist()[:30]
    neutral_prompts = pd.read_csv(neutral_csv)["prompt"].tolist()[:500]
    if extra_negative_csvs:
        for nc in extra_negative_csvs:
            neutral_prompts += pd.read_csv(nc)["prompt"].tolist()[:30]

    def _score(p):
        # Token-level max-sim: any (prompt_token × anchor_token) cos-sim
        h = pool_prompt_context(
            text_encoder, text_embedding, p, device, global_mean=global_mean,
        )  # [L_p, 4096]
        sim = h @ h_anchor.t().to(h.device)   # [L_p, L_c]
        return float(sim.max())

    s_unsafe = [_score(p) for p in unsafe_prompts]
    s_neutral = [_score(p) for p in neutral_prompts]

    # Use median-based gap as primary criterion; min/max kept in diag for inspection.
    import numpy as np
    med_gap = float(np.median(s_unsafe) - np.median(s_neutral))
    minmax_gap = min(s_unsafe) - max(s_neutral)

    if med_gap > 0.10:
        # Plenty of separation: pick τ at the boundary, sharp T.
        tau = (max(s_neutral) + min(s_unsafe)) / 2 if minmax_gap > 0 else float(np.median(s_neutral))
        T_ = max(med_gap / 4, 0.02)
    elif med_gap > 0.02:
        # Marginal separation: τ at neutral 90th percentile, looser T.
        tau = float(np.percentile(s_neutral, 90))
        T_ = 0.05
    else:
        print(
            f"WARN: concept {concept} has weak separation (med_gap={med_gap:.3f}); "
            "using fallback τ=0.30 T=0.10 — routing for this concept will be near-random"
        )
        tau, T_ = 0.30, 0.10
    return (
        h_anchor.cpu(),
        tau,
        T_,
        {"unsafe": s_unsafe, "neutral": s_neutral, "gap": minmax_gap, "med_gap": med_gap},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--task", default="t2v-1.3B")
    ap.add_argument(
        "--concepts_json",
        required=True,
        help='JSON: {"<c_name>": "<unsafe_csv_path>", ...}',
    )
    ap.add_argument("--neutral_csv", default="prompts/neutral/neutral_motion.csv")
    ap.add_argument("--output", default="ckpt/routing_config.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--center", action="store_true",
                    help="Apply global-mean centering before scoring. Default off "
                         "for token-level max-sim, which works better uncentered "
                         "(see ablation log A1.6).")
    args = ap.parse_args()
    args.no_center = not args.center  # internal flag retained for symmetry

    cfg = WAN_CONFIGS[args.task]
    device = torch.device(args.device)

    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device("cpu"),
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer),
    )
    text_encoder.model.to(device).eval()

    model = WanModel.from_pretrained(args.ckpt_dir).to(device).eval()
    text_embedding = model.text_embedding

    concepts = json.loads(Path(args.concepts_json).read_text())

    if args.no_center:
        global_mean = None
        print("Centering disabled (--no_center)")
    else:
        print("Estimating global mean from neutral prompts for centering...")
        neutral_for_mean = pd.read_csv(args.neutral_csv)["prompt"].tolist()[:200]
        global_mean = estimate_global_mean(text_encoder, neutral_for_mean, device)
        print(f"Global mean: ||μ||={global_mean.norm().item():.3f}")

    out = {
        "anchors": {}, "tau": {}, "T": {}, "diag": {},
        "global_mean": global_mean.cpu() if global_mean is not None else None,
        "centered": global_mean is not None,
    }
    concept_list = list(concepts.items())
    for c, unsafe_csv in concept_list:
        # Hard negatives = other concepts' unsafe prompts. Critical for
        # multi-concept routing; without them, gates leak across concepts.
        extra_negs = [u for c2, u in concept_list if c2 != c]
        h, tau, T_, diag = calibrate_one(
            c, unsafe_csv, args.neutral_csv,
            text_encoder, text_embedding, device,
            global_mean=global_mean,
            extra_negative_csvs=extra_negs,
        )
        out["anchors"][c] = h
        out["tau"][c] = tau
        out["T"][c] = T_
        out["diag"][c] = diag
        print(
            f"[{c}] tau={tau:.3f} T={T_:.3f} "
            f"med_gap={diag['med_gap']:+.3f} minmax_gap={diag['gap']:+.3f}"
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"Saved routing config to {args.output}")


if __name__ == "__main__":
    main()
