# Design notes

This document collects the structural reasons behind the two most important
architectural choices in FADE:

1. why the Concept Redirect (CR) closed-form edit targets the cross-attention
   K/V projections of the Wan2.1 DiT, and not the FFN, as adopted in some
   image-domain and dual-stream video refusal-vector methods;
2. why concept references are sampled in the post-MLP context space rather
   than in the raw T5 hidden space.

The paper's Appendix `app:edit-locus` summarises the same argument in
condensed form. The version here is longer and code-cross-referenced.

## 1. Single-stream vs. dual-stream DiTs

Some prior work on video concept erasure (notably the closed-form refusal-vector
edit of Facchiano et al., 2025) targets the FFN of MMDiT-style backbones such
as Open-Sora 2.0. Those backbones are dual-stream: each block contains parallel
text-FFN and image-FFN modules whose inputs are restricted to the text or
image stream respectively. Editing the text-FFN then acts on pure-text
activations, in a hidden space that matches the refusal vector `r` extracted
from the same position.

Wan2.1 is a pure single-stream DiT. Every block runs:

1. self-attention over ~7800 video tokens,
2. cross-attention with text supplied as frozen K/V,
3. a single FFN consuming a mixture of self- and cross-attention residuals.

There is no text-native FFN in Wan2.1. Any closed-form edit targeting the
single FFN

- acts on activations dominated by the video stream, and
- perturbs spatio-temporal modelling on every prompt, including neutral ones.

In our pilot experiments, editing the FFN reproduced two failure modes:
over-sharpened static textures and motion damage on neutral motion prompts.
Both are directly explained by the video-stream dominance of the FFN's input.

## 2. Cross-attention K/V is the unique text-to-video bottleneck

In Wan2.1 (see `wan/modules/model.py`), text influences generation only via
each block's cross-attention `.k` / `.v` linear maps. The post-text-embedding
context

    ctx = model.text_embedding(text_encoder(prompt))

is consumed only there. Editing `blocks[l].cross_attn.k` and
`blocks[l].cross_attn.v` therefore cuts the concept signal at its single
point of entry into the video stream, without touching the FFN, so motion
fidelity on prompts that route around the edited concept is preserved.

Closed-form editing in the post-MLP context space is well-conditioned: the
K/V projections are linear maps from that space, and concept references
obtained as MPE-averaged post-MLP embeddings live in the same space.

## 3. Why concept references sit in post-MLP space, not raw T5 space

A natural alternative is to extract the refusal direction in the raw T5
hidden space (4096-d, umT5-XXL) and project it into the DiT context space
when applying the edit. This fails because `model.text_embedding` contains a
GELU non-linearity, so directions in T5 space do not map linearly into
context space. The closed-form edit equation

    W' = W (I - λ · U · r̂r̂ᵀ · Uᵀ / ‖r̂‖²)

becomes ill-defined when `r̂` is drawn from a space related to `W`'s input by
a non-linear map.

Sampling concept references at the post-MLP last-content-token position, as
the CR solver does in `cr_edit.py`, is the smallest change that restores
linearity and matches the K/V projections' input space.

## 4. Alternative edit sites we considered

Before settling on cross-attention K/V, three other sites were evaluated:

- **Video-side query `W_Q`.** Not concept-specific; editing it corrupts every
  cross-attention head on every prompt, including neutral ones.
- **FFN first linear `W_ffn0`.** Matches Facchiano et al.'s locus, but as
  above, the input is a mixed self/cross-attn residual. CR ablations confirm
  that the same solver applied at `W_ffn0` cannot match motion smoothness on
  neutral prompts.
- **Self-attention projections.** Similarly text-blind.

None of the three provides the surgical text-to-video isolation that K/V
editing offers, which is why CR retains K/V as its sole edit locus and the
residual frame-reactivation gap is closed by FAE (a parametric correction on
the FFN path) rather than by another closed-form edit.

## 5. Wan2.1 text-conditioning dimensions

For reference:

| Variant     | `dim` | `num_blocks` |
|-------------|-------|--------------|
| Wan2.1-T2V-1.3B | 1536 | 30 |
| Wan2.1-T2V-14B  | 5120 | 40 |

CR and FAE were validated on the 1.3B variant. The 14B numbers reported in
the paper appendix use the same code paths with no architectural changes.
