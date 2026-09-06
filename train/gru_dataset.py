"""GRU data pipeline: RAM store of the feature caches + prefix-cut crop
augmentation + length-bucketed batch assembly.

Crop re-anchoring: a crop starting at grid node j takes tvt_true[j] as its
pseudo-anchor; equivalently the anchor-relative channel and the target are
rebased by their value at j.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gru_features import ANCHOR_CH, CH_IDX, CONFIG as FCFG

FEAT_DIR = Path(__file__).resolve().parent / "artifacts" / "feat"
TS = FCFG["tvt_scale"]; MS = FCFG["md_scale"]; CLIP = FCFG["clip"]

AUG = dict(crops_per_well=2, crop_max_frac=0.5, crop_min_tail_ft=800.0,
           gr_jitter=0.05, ch_drop=0.1)
JITTER_CH = [CH_IDX[n] for n in ("gr_norm", "gr_smooth", "gr_rstd")]
DROP_CH = [CH_IDX[n] for n in (
    "gr_rstd", "gr_gap", "z_curv", "node_dip", "ir",
    "look_slope250", "look_slope1000", "look_gr_mean", "look_gr_std",
    "fam_sp_blate", "fam_beam_mean", "fam_poly1_t500", "fam_flat")]


class WellStore:
    """All training wells in RAM."""

    def __init__(self, wids):
        self.wells = {}
        for w in wids:
            with np.load(FEAT_DIR / f"{w}.npz") as f:
                self.wells[w] = {k: f[k].astype(np.float32) for k in f.files}
        self.n_ch = next(iter(self.wells.values()))["X"].shape[0]

    def full_sample(self, wid):
        d = self.wells[wid]
        return d["X"], d["y_g"]

    def crop_sample(self, wid, j):
        """Prefix-cut crop starting at grid node j, re-anchored at j."""
        d = self.wells[wid]
        X = d["X"][:, j:].copy()
        md_g = d["md_g"][j:]
        Lc = X.shape[1]
        # rebase the stored channel itself (clip is inert at these magnitudes)
        X[ANCHOR_CH] = np.clip(X[ANCHOR_CH] - X[ANCHOR_CH][0], -CLIP, CLIP)
        X[CH_IDX["md_since"]] = (md_g - md_g[0]) / MS
        X[CH_IDX["idx_norm"]] = np.arange(Lc, dtype=np.float32) / max(Lc - 1, 1)
        X[CH_IDX["total_len"]] = (md_g[-1] - md_g[0]) / MS
        y = d["y_g"][j:] - d["y_g"][j]
        return X, y

    def crop_starts(self, wid, rng, n):
        d = self.wells[wid]
        L = d["X"].shape[1]
        jmax = min(int(L * AUG["crop_max_frac"]),
                   L - int(AUG["crop_min_tail_ft"] / FCFG["grid_step"]))
        if jmax <= 1:
            return []
        return list(rng.integers(1, jmax, size=n))


def epoch_items(store, wids, rng, use_crops):
    items = [(w, 0) for w in wids]
    if use_crops:
        for w in wids:
            items += [(w, int(j)) for j in
                      store.crop_starts(w, rng, AUG["crops_per_well"])]
    rng.shuffle(items)
    return items


def batches(store, items, batch_size, rng, train_aug):
    """Length-bucketed padded batches. Yields (X (B,C,L), y (B,L), mask (B,L), lengths)."""
    samples = []
    for w, j in items:
        X, y = store.full_sample(w) if j == 0 else store.crop_sample(w, j)
        samples.append((X, y))
    order = np.argsort([s[0].shape[1] for s in samples], kind="stable")
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(chunks)
    for ch in chunks:
        Ls = [samples[i][0].shape[1] for i in ch]
        Lm = max(Ls)
        B = len(ch)
        Xb = np.zeros((B, store.n_ch, Lm), np.float32)
        yb = np.zeros((B, Lm), np.float32)
        mb = np.zeros((B, Lm), np.float32)
        for bi, i in enumerate(ch):
            X, y = samples[i]
            L = X.shape[1]
            Xb[bi, :, :L] = X; yb[bi, :L] = y; mb[bi, :L] = 1.0
        if train_aug:
            Xb[:, JITTER_CH, :] += rng.normal(
                0.0, AUG["gr_jitter"], size=(B, len(JITTER_CH), Lm)).astype(np.float32)
            drop = rng.random((B, len(DROP_CH))) < AUG["ch_drop"]
            for bi in range(B):
                for di, chn in enumerate(DROP_CH):
                    if drop[bi, di]:
                        Xb[bi, chn, :] = 0.0
        yield Xb, yb, mb, np.asarray(Ls, np.int64)
