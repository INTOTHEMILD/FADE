#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Concept Redirect (CR) — Phase 1 of FADE.

Closed-form cross-attention K/V edit on the Wan2.1 DiT. The solver follows
the UCE recipe (Gandikota et al., WACV 2024, arXiv:2308.14761): for every
WanAttentionBlock we replace
    blocks.{i}.cross_attn.k.weight
    blocks.{i}.cross_attn.v.weight
with the CR-edited version; nothing else in the network is touched.

Concept representation
----------------------
We re-use SD-UCE's "last content token" convention:
    t5_emb = T5EncoderModel([concept])        # [L, 4096] (trimmed, umT5 appends </s>)
    c_i    = model.text_embedding(t5_emb)[L-2]  # the cross-attn context at that token
c_i is the vector that cross_attn.k / cross_attn.v actually see, so it is the
natural representation for the closed-form edit. (cross_attn.norm_k is applied
AFTER self.k; we still only modify the linear — the edit stays correct in
expectation because RMSNorm is a positionwise rescaling.)

Input data format
-----------------
This script consumes ONLY plain concept strings — no prompt CSV, no video,
no DiT forward. Every signal comes from T5 + model.text_embedding.

Concepts are `;`-separated strings:
    --edit_concepts "church"                      (single)
    --edit_concepts "church;cathedral;temple"     (multi)

Three roles per run:
    edit_concepts     c_erase    : concepts whose K/V output should be rewritten
    guide_concepts    target     : where to redirect to (W_old @ c_guide is
                                   the desired new output for c_erase).
                                   Default '' for object (erase-to-nothing),
                                   'art' for art (erase-to-generic-art).
                                   Length must equal edit_concepts OR be 1
                                   (broadcast to every edit).
    preserve_concepts anchors    : concepts whose W·c output must stay put.
                                   Free-length list, decoupled from edits.

Three category presets (--concept_type):
    object : animals / objects / NSFW persons.    default guide ''
    art    : artist names / styles.               default guide 'art'
(Only affects the default guide value and the templates used by
--expand_prompts.)

Output
------
A full drop-in Wan checkpoint directory at `<save_dir>/<exp_name>/`:
  - `diffusion_pytorch_model.safetensors` — the edited DiT weights.
  - `config.json` — the DiT config.
  - `models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.1_VAE.pth`, tokenizer files —
    symlinked (default) or copied from the source ckpt.
  - `uce_meta.json` — audit record of concepts, scales, and per-projection
    edit magnitudes.

The directory is usable as `--ckpt_dir` for `generate.py` and `train_fae.py`
with no other changes.

Example
-------
  python cr_edit.py \
    --task t2v-1.3B \
    --ckpt_dir ckpt/Wan2.1-T2V-1.3B \
    --edit_concepts "church" \
    --concept_type object \
    --preserve_concepts "fortress;cottage;castle;cabin;school" \
    --lamb 0.5 --erase_scale 1.0 --preserve_scale 0.1 \
    --save_dir ckpt --exp_name Wan2.1-T2V-1.3B-cr-church \
    2>&1 | tee logs/cr_church.log
"""
import argparse
import gc
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch

# Allow running the script directly from the Wan2.1 directory.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        prog='cr_edit',
        description='Concept Redirect (CR) closed-form K/V edit for Wan2.1 T2V')
    p.add_argument('--task', type=str, default='t2v-1.3B',
                   choices=list(WAN_CONFIGS.keys()))
    p.add_argument('--ckpt_dir', type=str, required=True,
                   help='Wan2.1 checkpoint directory (contains config.json, '
                        'T5 ckpt, VAE ckpt, etc.)')
    p.add_argument('--device', type=str, default='cuda:0')

    # Single: 'church'.  Multi: 'church;cathedral;temple'.  Whitespace stripped.
    p.add_argument('--edit_concepts', type=str, required=True,
                   help='Concepts to erase, separated by ;')
    # Same `;` form. Length must equal edit_concepts, OR be exactly 1
    # (broadcast). Default: '' for object, 'art' for art — see module docstring.
    p.add_argument('--guide_concepts', type=str, default=None,
                   help='Where to redirect the erased concepts. Separated by ;. '
                        'If given a single item it is broadcast to every '
                        'edit_concept. Defaults to "" (object) / "art" (art).')
    # Anchors retained at their original W·c output. Free-length, can be empty;
    # decoupled from edit_concepts (e.g. erase 'church', preserve 'castle;cabin').
    p.add_argument('--preserve_concepts', type=str, default=None,
                   help='Concepts to preserve, separated by ;')
    # Only switches default guide + prompt templates — does NOT change the math.
    p.add_argument('--concept_type', type=str, required=True,
                   choices=['art', 'object'])

    p.add_argument('--erase_scale', type=float, default=1.0)
    p.add_argument('--preserve_scale', type=float, default=1.0)
    p.add_argument('--lamb', type=float, default=0.5,
                   help='L2 regularisation weight (λ in Eq. 7 of the paper)')

    p.add_argument('--expand_prompts', type=str, default='false',
                   choices=['true', 'false'],
                   help='If true, augment each concept with a few prompt '
                        'templates ("photo of X", "image of X", ...).')

    # Video-native extension: replace each bare concept anchor with the set of
    # full unsafe prompts from a CSV. closed-form math unchanged; rank of the
    # erase term grows from 1 to N_prompts per concept, so the edit operates
    # on the manifold of *contextualised* concept appearances rather than the
    # bare lexical embedding. Mutually exclusive with --expand_prompts=true.
    p.add_argument('--edit_prompts_csvs', type=str, default=None,
                   help='Per-concept prompt CSVs separated by ;. Same length '
                        'as --edit_concepts. Each CSV must have a "prompt" '
                        'column. When given, the closed-form erase set is '
                        'built from these prompts instead of bare concept '
                        'words. guide_concepts and preserve_concepts remain '
                        'bare strings.')

    # Video-native extension #2: replace each bare concept anchor with the mean
    # of post-MLP embeddings over the concept itself plus a small set of motion-
    # grounded variants (e.g. "parachute drifting", "parachute deploying"). This
    # rotates c_erase toward the manifold of contextualised motion appearances
    # without inflating the rank of the erase term (still rank-1 per concept).
    # Only activates the modifiers listed for each edit concept; concepts absent
    # from the JSON keep their bare-token anchor.
    p.add_argument('--motion_modifiers_json', type=str, default=None,
                   help='Path to JSON {concept: [variant1, variant2, ...]}. '
                        'When given, each edit concept c with a key in the JSON '
                        'has its c_erase replaced by the mean of '
                        '{c, variant_1, ..., variant_N} post-MLP last-token '
                        'embeddings. Mutually exclusive with --edit_prompts_csvs.')

    p.add_argument('--save_dir', type=str, default='./uce_models')
    p.add_argument('--exp_name', type=str, default=None)
    p.add_argument('--link_mode', type=str, default='symlink',
                   choices=['symlink', 'copy'],
                   help='How to carry T5/VAE/tokenizer from ckpt_dir into the '
                        'output ckpt dir. symlink is cheap; copy is portable.')
    return p.parse_args()


def save_full_ckpt(model, src_ckpt_dir, output_dir, link_mode='symlink'):
    """Write edited DiT weights and symlink/copy T5/VAE/config into output_dir.

    Produces a drop-in Wan checkpoint dir usable by generate.py."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))

    src = Path(src_ckpt_dir)
    dit_owned = {'config.json',
                 'diffusion_pytorch_model.safetensors',
                 'diffusion_pytorch_model.safetensors.index.json'}
    for name in os.listdir(src):
        dst = out / name
        if dst.exists():
            continue
        src_path = src / name
        if name in dit_owned:
            continue
        if name.startswith('diffusion_pytorch_model') and (
                name.endswith('.safetensors') or name.endswith('.bin')):
            continue
        if link_mode == 'symlink':
            os.symlink(src_path.resolve(), dst)
        elif link_mode == 'copy':
            if src_path.is_dir():
                shutil.copytree(src_path, dst)
            else:
                shutil.copy2(src_path, dst)


def _split(s):
    # 'a;b ; c' → ['a', 'b', 'c'].  None/'' → [].  Empty tokens dropped.
    if s is None:
        return []
    return [x.strip() for x in s.split(';') if x.strip()]


def _expand(edits, guides, concept_type):
    # Augment each (edit, guide) pair with prompt templates so the same concept
    # is erased in multiple surface forms (e.g. "photo of cat", "painting of cat").
    # Effective concept count multiplies by 1 + len(tpls) = 6.
    tpls = (['painting by {}', 'art by {}', 'artwork by {}',
             'picture by {}', 'style of {}']
            if concept_type == 'art'
            else ['image of {}', 'photo of {}', 'portrait of {}',
                  'picture of {}', 'painting of {}'])
    out_e, out_g = list(edits), list(guides)
    for e, g in zip(edits, guides):
        for t in tpls:
            out_e.append(t.format(e))
            # Empty guide stays empty — don't expand '' into 'photo of '.
            out_g.append(t.format(g) if g else g)
    return out_e, out_g


# ----------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------
@torch.no_grad()
def encode_concept_vector(text_encoder, text_embedding, concept, device):
    """
    Run T5 + WanModel.text_embedding and return the last content-token vector.
    This is the `c_i` that cross_attn.{k,v} see for this concept.

    Returns:
        [dim] tensor on `device` in text_embedding's dtype.
    """
    # T5EncoderModel returns a list of trimmed [L, 4096] tensors (mask applied).
    t5_emb = text_encoder([concept], device)[0]
    # umT5 w/ add_special_tokens=True appends </s>; skip it (matches SD-UCE's
    # `attention_mask.sum() - 2` convention).
    last_idx = max(t5_emb.size(0) - 2, 0)
    token = t5_emb[last_idx:last_idx + 1]  # [1, 4096]

    te_dtype = next(text_embedding.parameters()).dtype
    ctx = text_embedding(token.to(te_dtype))  # [1, dim]
    return ctx[0].detach()


@torch.no_grad()
def encode_motion_grounded_vector(text_encoder, text_embedding, concept,
                                  variants, device):
    """Mean of last-token post-MLP embeddings over {concept, variants…}.

    Encodes every string in [concept, *variants] separately, takes each
    string's last content-token embedding (T5 → text_embedding MLP), and
    averages. Result is a single [dim] vector that occupies the same role
    as encode_concept_vector but is rotated toward the manifold of
    contextualised motion appearances.
    """
    bag = [concept] + list(variants)
    embs = [encode_concept_vector(text_encoder, text_embedding, s, device)
            for s in bag]
    return torch.stack(embs, dim=0).mean(dim=0).detach()


def uce_closed_form(w_old,
                    c_erase, v_guide,
                    c_preserve, v_preserve,
                    erase_scale, preserve_scale, lamb,
                    device):
    """
    Closed-form UCE update, Eq. 7 of the paper.

    Args:
        w_old: [dim_out, dim_in] original Linear weight.
        c_erase, c_preserve: lists of [dim_in] context vectors.
        v_guide, v_preserve: lists of [dim_out] target outputs (already
            equal to W_old @ c for the corresponding concept).

    Returns:
        [dim_out, dim_in] edited weight, cast back to w_old.dtype.
    """
    out_dtype = w_old.dtype
    w = w_old.to(torch.float32).to(device)
    dim_in = w.shape[1]

    mat1 = lamb * w
    mat2 = lamb * torch.eye(dim_in, device=device, dtype=torch.float32)

    for c, v in zip(c_erase, v_guide):
        c = c.to(torch.float32).to(device).unsqueeze(1)    # [dim_in, 1]
        v = v.to(torch.float32).to(device).unsqueeze(1)    # [dim_out, 1]
        mat1 += erase_scale * (v @ c.T)
        mat2 += erase_scale * (c @ c.T)

    for c, v in zip(c_preserve, v_preserve):
        c = c.to(torch.float32).to(device).unsqueeze(1)
        v = v.to(torch.float32).to(device).unsqueeze(1)
        mat1 += preserve_scale * (v @ c.T)
        mat2 += preserve_scale * (c @ c.T)

    w_new = mat1 @ torch.inverse(mat2)
    return w_new.to(out_dtype)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(stream=sys.stdout)])

    device = torch.device(args.device)
    cfg = WAN_CONFIGS[args.task]

    # ---- parse concept lists ------------------------------------------------
    # Multi-concept rules (this block enforces them):
    #   1) guide omitted          → category default ('' object / 'art' art)
    #   2) 1 guide,  N edits      → broadcast to all edits
    #   3) N guides, N edits      → paired by index
    #   4) anything else          → ValueError (length mismatch)
    edit_concepts = _split(args.edit_concepts)
    guide_concepts = _split(args.guide_concepts)
    if not guide_concepts:
        guide_concepts = ['art'] if args.concept_type == 'art' else ['']
    if len(guide_concepts) == 1 and len(edit_concepts) > 1:
        guide_concepts = guide_concepts * len(edit_concepts)
    if len(guide_concepts) != len(edit_concepts):
        raise ValueError(
            f'edit_concepts ({len(edit_concepts)}) and guide_concepts '
            f'({len(guide_concepts)}) must have equal length — separate with ;')
    # Preserve list is independent of edits — order/length need not match.
    preserve_concepts = _split(args.preserve_concepts)

    if args.expand_prompts == 'true':
        edit_concepts, guide_concepts = _expand(
            edit_concepts, guide_concepts, args.concept_type)

    # CSV-driven prompt expansion: each concept's bare string is replaced by
    # all rows of its unsafe-prompt CSV; the same guide is broadcast across
    # all rows of one concept. Length-validated against the original (pre-
    # template-expand) edit_concepts.
    if args.edit_prompts_csvs:
        csvs = _split(args.edit_prompts_csvs)
        if len(csvs) != len(_split(args.edit_concepts)):
            raise ValueError(
                f'edit_prompts_csvs ({len(csvs)}) must equal the number of '
                f'original edit_concepts ({len(_split(args.edit_concepts))}).')
        original_edits = _split(args.edit_concepts)
        original_guides = list(_split(args.guide_concepts) or [])
        if not original_guides:
            original_guides = ['art' if args.concept_type == 'art' else ''] * len(original_edits)
        if len(original_guides) == 1 and len(original_edits) > 1:
            original_guides = original_guides * len(original_edits)

        new_edits, new_guides = [], []
        per_concept_counts = []
        for ce, cg, csv_path in zip(original_edits, original_guides, csvs):
            df = pd.read_csv(csv_path)
            prompts = [str(p).strip() for p in df['prompt'].tolist() if str(p).strip()]
            new_edits.extend(prompts)
            new_guides.extend([cg] * len(prompts))
            per_concept_counts.append((ce, len(prompts)))
        edit_concepts = new_edits
        guide_concepts = new_guides
        for ce, n in per_concept_counts:
            logging.info(f'  {ce}: {n} prompt rows')

    logging.info(f'Erasing    : {len(edit_concepts)} item(s)')
    logging.info(f'Guiding    : {len(guide_concepts)} item(s) ({set(guide_concepts) if guide_concepts else "-"})')
    logging.info(f'Preserving : {preserve_concepts}')

    os.makedirs(args.save_dir, exist_ok=True)
    exp_name = args.exp_name or 'uce_wan21_test'
    out_dir = os.path.join(args.save_dir, exp_name)

    # ---- load T5 ------------------------------------------------------------
    logging.info('Loading T5 (umT5-XXL) text encoder...')
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.ckpt_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.ckpt_dir, cfg.t5_tokenizer),
    )
    text_encoder.model.to(device).eval()

    # ---- load WanModel (we only need text_embedding + cross_attn k/v) -------
    logging.info(f'Loading WanModel from {args.ckpt_dir}')
    model = WanModel.from_pretrained(args.ckpt_dir)
    model.eval().requires_grad_(False)
    model.to(device)

    # ---- encode every unique concept to its last-content-token vector ------
    all_concepts = list(dict.fromkeys(
        edit_concepts + guide_concepts + preserve_concepts))
    logging.info(f'Encoding {len(all_concepts)} unique concepts via T5 + text_embedding')
    concept_c = {}
    for c in all_concepts:
        concept_c[c] = encode_concept_vector(
            text_encoder, model.text_embedding, c, device)

    # Motion-grounded anchors override bare-token c_e for listed edit concepts.
    if args.motion_modifiers_json:
        with open(args.motion_modifiers_json) as f:
            motion_modifiers = json.load(f)
        n_overridden = 0
        for ce in set(edit_concepts):
            if ce in motion_modifiers:
                variants = motion_modifiers[ce]
                concept_c[ce] = encode_motion_grounded_vector(
                    text_encoder, model.text_embedding, ce, variants, device)
                n_overridden += 1
                logging.info(
                    f'  motion anchor [{ce}] := mean over '
                    f'{1 + len(variants)} variants')
        logging.info(
            f'Motion-grounded anchors applied to {n_overridden}/'
            f'{len(set(edit_concepts))} edit concepts')

    # T5 is no longer needed — free it before running the UCE linear algebra.
    del text_encoder
    torch.cuda.empty_cache()
    gc.collect()

    # ---- collect all cross_attn k / v modules -------------------------------
    uce_module_names = [
        name for name, _ in model.named_modules()
        if name.endswith('cross_attn.k') or name.endswith('cross_attn.v')
    ]
    logging.info(f'Editing {len(uce_module_names)} cross-attn projections '
                 f'(expected 2 × num_layers = {2 * cfg.num_layers})')

    # ---- per-module closed-form update --------------------------------------
    start = time.time()
    delta_ratios = []
    for name in uce_module_names:
        mod = model.get_submodule(name)
        w_old = mod.weight.detach()
        w_old_f32 = w_old.to(torch.float32)

        # Erase target: W_new @ c_erase ≈ W_old @ c_guide  (redirect c_erase
        # to whatever the original model produced for c_guide). Empty guide ''
        # encodes "no concept", i.e. erase-to-nothing.
        v_guide_list = [w_old_f32 @ concept_c[g].to(torch.float32)
                        for g in guide_concepts]
        # Preserve target: W_new @ c_preserve ≈ W_old @ c_preserve (identity).
        v_preserve_list = [w_old_f32 @ concept_c[p].to(torch.float32)
                           for p in preserve_concepts]

        w_new = uce_closed_form(
            w_old=w_old,
            c_erase=[concept_c[e] for e in edit_concepts],
            v_guide=v_guide_list,
            c_preserve=[concept_c[p] for p in preserve_concepts],
            v_preserve=v_preserve_list,
            erase_scale=args.erase_scale,
            preserve_scale=args.preserve_scale,
            lamb=args.lamb,
            device=device,
        )

        delta = (w_new.to(torch.float32).to(device)
                 - w_old_f32.to(device)).norm().item()
        base = w_old_f32.norm().item()
        ratio = delta / max(base, 1e-12)
        delta_ratios.append(ratio)
        logging.info(f'  {name}: ||ΔW||/||W|| = {ratio:.4%}')

        mod.weight.data.copy_(
            w_new.to(mod.weight.dtype).to(mod.weight.device))

    mean_ratio = sum(delta_ratios) / max(len(delta_ratios), 1)
    max_ratio = max(delta_ratios) if delta_ratios else 0.0
    logging.info(f'Edit magnitude: mean {mean_ratio:.4%}, max {max_ratio:.4%} '
                 f'over {len(delta_ratios)} projections')

    # ---- save full Wan ckpt dir (drop-in for generate.py) -------------------
    save_full_ckpt(model, args.ckpt_dir, out_dir, link_mode=args.link_mode)

    meta = {
        'task': args.task,
        'src_ckpt_dir': args.ckpt_dir,
        'edit_concepts': edit_concepts,
        'guide_concepts': guide_concepts,
        'preserve_concepts': preserve_concepts,
        'erase_scale': args.erase_scale,
        'preserve_scale': args.preserve_scale,
        'lamb': args.lamb,
        'expand_prompts': args.expand_prompts,
        'n_edit': len(edit_concepts),
        'n_preserve': len(preserve_concepts),
        'mean_delta_ratio': mean_ratio,
        'max_delta_ratio': max_ratio,
        'method': 'UCE-Wan2.1-v1',
    }
    with open(Path(out_dir) / 'uce_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    logging.info(f'CR edit done in {time.time() - start:.1f}s')
    logging.info(f'Edited {len(uce_module_names)} K/V projections → {out_dir}')


if __name__ == '__main__':
    main()
