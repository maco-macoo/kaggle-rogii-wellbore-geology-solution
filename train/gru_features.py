"""GRU feature builder: PF dump dict -> 4ft-grid sequence tensors (36 channels)
for the bidirectional-GRU tail refiner. Leak rules:

  * build_features() DOES NOT take the truth (enforced by a canary at training).
  * All normalization is per-well with fixed constants (25ft / gr_sigma / 1000ft);
    no scaler is ever fit across wells.
  * The only raw-data dependency is the GR NaN gap mask (the dump's gr_ev is
    already interpolated); eval-zone GR observability is legal at inference.

The same module text is used at training time (train/gru_features.py) and
embedded in the submission kernel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = (Path("/kaggle/input/rogii-wellbore-geology-prediction")
             if Path("/kaggle").exists() else ROOT / "input") / "train"
ART = Path(__file__).resolve().parent / "artifacts"
DUMP_DIR = ART / "dump"
FEAT_DIR = ART / "feat"

CONFIG = dict(
    grid_step=4.0, tvt_scale=25.0, md_scale=1000.0, clip=8.0,
    lik_scale=5.0,                       # pfens_s5 weighting (same as the PF ensemble)
    gr_smooth_ft=25.0, gr_rstd_ft=50.0,
    look_gr_ft=500.0, look_slope_fts=(250.0, 1000.0), dip_trail_ft=200.0,
    grmm_offsets=(-10.0, 0.0, 10.0), frac_band_ft=5.0,
    n_jobs=14, seed=42,
)

# channel registry: (name, group) -- group "cand" = candidate/PF-derived,
# "look" = legal look-ahead, "" = own-well GR/trajectory/position.
CHANNELS = [
    ("pf_anchor_rel", "cand"), ("pf_s3_dis", "cand"), ("pf_s8_dis", "cand"),
    ("pf_s12_dis", "cand"), ("pf_wstd", "cand"), ("pf_q90_10", "cand"),
    ("pf_q75_25", "cand"), ("pf_top1_dis", "cand"), ("pf_top8_dis", "cand"),
    ("pf_frac5", "cand"), ("pf_grmm_m10", "cand"), ("pf_grmm_0", "cand"),
    ("pf_grmm_p10", "cand"), ("pf_ess", "cand"),
    ("gr_norm", ""), ("gr_smooth", ""), ("gr_rstd", ""), ("gr_gap", ""),
    ("z_rel", ""), ("z_slope", ""), ("z_curv", ""),
    ("z_remain", "look"), ("look_slope250", "look"), ("look_slope1000", "look"),
    ("look_gr_mean", "look"), ("look_gr_std", "look"),
    ("node_dip", "cand"), ("ir", ""),
    ("md_since", ""), ("dist_end", ""), ("idx_norm", ""), ("total_len", ""),
    ("fam_sp_blate", "cand"), ("fam_beam_mean", "cand"),
    ("fam_poly1_t500", "cand"), ("fam_flat", "cand"),
]
CH_IDX = {n: i for i, (n, _) in enumerate(CHANNELS)}
CAND_CH = [i for i, (_, g) in enumerate(CHANNELS) if g == "cand"]
LOOK_CH = [i for i, (_, g) in enumerate(CHANNELS) if g == "look"]
ANCHOR_CH = CH_IDX["pf_anchor_rel"]        # crop re-anchoring rebases this channel
N_CH = len(CHANNELS)


def _fwd_mean(v, k):
    """out[i] = mean(v[i:i+k]) with shrinking window at the tail."""
    n = len(v)
    c = np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])
    hi = np.minimum(np.arange(n) + k, n)
    cnt = hi - np.arange(n)
    return (c[hi] - c[:-1]) / np.maximum(cnt, 1)


def _fwd_std(v, k):
    m1 = _fwd_mean(v, k)
    m2 = _fwd_mean(v * v, k)
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))


def _smooth(v, k):
    """centered boxcar mean, width k samples (edge-padded)."""
    if k <= 1:
        return v.astype(np.float64)
    pad = k // 2
    vp = np.pad(v.astype(np.float64), pad, mode="edge")
    c = np.concatenate([[0.0], np.cumsum(vp)])
    out = (c[k:] - c[:-k]) / k
    return out[: len(v)]


def build_features(d, gr_raw_nan):
    """d: dict of dump arrays WITHOUT tvt_true. gr_raw_nan: raw eval-zone GR (NaN=gap).

    Returns X (C,L) float32, md_g (L,), base_g (L,) float64.
    """
    cfg = CONFIG
    names = [str(n) for n in d["names"]]
    ni = {n: i for i, n in enumerate(names)}
    cands = d["cands"].astype(np.float64)
    md = d["md_ev"].astype(np.float64)
    z = d["z_ev"].astype(np.float64)
    gr = d["gr_ev"].astype(np.float64)
    liks = d["liks"].astype(np.float64)
    gg = d["gg"].astype(np.float64)
    gmin = float(d["gmin"]); gstep = float(d["gstep"])
    last_tvt = float(d["last_tvt"]); last_md = float(d["last_md"])
    ir = float(d["ir"]); gr_sigma = float(d["gr_sigma"])

    base = cands[ni["pfens_s5"]]
    seeds = cands[[ni[f"pf{s:03d}"] for s in range(len(liks))]]     # (S, n_ev)
    w = np.exp((liks - liks.max()) / cfg["lik_scale"]); w /= w.sum()

    # ---- 4ft grid ----
    step = cfg["grid_step"]
    md_g = np.arange(md[0], md[-1], step)
    if len(md_g) == 0 or md[-1] - md_g[-1] > 0.5:
        md_g = np.append(md_g, md[-1])
    L = len(md_g)

    def lin(v):
        return np.interp(md_g, md, v)

    med_dmd = float(np.median(np.diff(md))) if len(md) > 1 else 1.0
    k_native = max(1, int(round(step / max(med_dmd, 1e-6))))

    base_g = lin(base)
    z_g = lin(z)
    gr_g = lin(_smooth(gr, k_native))                # anti-aliased GR on grid
    gr_med = float(np.median(gr))

    ts, ms, cl = cfg["tvt_scale"], cfg["md_scale"], cfg["clip"]
    X = np.zeros((N_CH, L), dtype=np.float64)

    def put(name, v):
        X[CH_IDX[name]] = v

    # ---- PF posterior block ----
    put("pf_anchor_rel", (base_g - last_tvt) / ts)
    for sc, nm in ((3, "pf_s3_dis"), (8, "pf_s8_dis"), (12, "pf_s12_dis")):
        put(nm, (lin(cands[ni[f"pfens_s{sc}"]]) - base_g) / ts)
    wstd = np.sqrt(np.maximum((w[:, None] * (seeds - base[None, :]) ** 2).sum(0), 0.0))
    put("pf_wstd", lin(wstd) / ts)
    q10, q25, q75, q90 = np.percentile(seeds, [10, 25, 75, 90], axis=0)
    put("pf_q90_10", lin(q90 - q10) / ts)
    put("pf_q75_25", lin(q75 - q25) / ts)
    put("pf_top1_dis", (lin(cands[ni["pf_top1lik"]]) - base_g) / ts)
    put("pf_top8_dis", (lin(cands[ni["pf_top8lik"]]) - base_g) / ts)
    put("pf_frac5", lin((np.abs(seeds - base[None, :]) <= cfg["frac_band_ft"]).mean(0)))
    tw_axis = gmin + gstep * np.arange(len(gg))
    for dlt, nm in zip(cfg["grmm_offsets"], ("pf_grmm_m10", "pf_grmm_0", "pf_grmm_p10")):
        mm = (gr - np.interp(base + dlt, tw_axis, gg)) / gr_sigma
        put(nm, lin(_smooth(mm, k_native)))
    put("pf_ess", np.full(L, 1.0 / (np.sum(w ** 2) * len(w))))

    # ---- GR block ----
    grn_native = (gr - gr_med) / gr_sigma
    put("gr_norm", (gr_g - gr_med) / gr_sigma)
    put("gr_smooth", lin(_smooth(grn_native, max(1, int(round(cfg["gr_smooth_ft"] / max(med_dmd, 1e-6)))))))
    k_rstd = max(2, int(round(cfg["gr_rstd_ft"] / max(med_dmd, 1e-6))))
    pad = k_rstd // 2
    gp = np.pad(grn_native, pad, mode="edge")
    c1 = np.concatenate([[0.0], np.cumsum(gp)])
    c2 = np.concatenate([[0.0], np.cumsum(gp * gp)])
    m1 = (c1[k_rstd:] - c1[:-k_rstd]) / k_rstd
    m2 = (c2[k_rstd:] - c2[:-k_rstd]) / k_rstd
    put("gr_rstd", lin(np.sqrt(np.maximum(m2 - m1 * m1, 0.0))[: len(gr)]))
    gap = np.isnan(gr_raw_nan).astype(np.float64)
    put("gr_gap", lin(_smooth(gap, k_native)))

    # ---- trajectory + legal look-ahead ----
    put("z_rel", (z_g - z_g[0]) / 100.0)
    slope_z = np.gradient(_smooth(z_g, 5), md_g)
    put("z_slope", slope_z * 20.0)
    put("z_curv", np.gradient(slope_z, md_g) * 2000.0)
    put("z_remain", (z_g[-1] - z_g) / 100.0)
    for W, nm in zip(cfg["look_slope_fts"], ("look_slope250", "look_slope1000")):
        k = max(2, int(round(W / step)))
        dz_f = _fwd_mean(np.gradient(z_g, md_g), k)
        put(nm, dz_f * 20.0)
    k_gr = max(2, int(round(cfg["look_gr_ft"] / step)))
    grn_g = (gr_g - gr_med) / gr_sigma
    put("look_gr_mean", _fwd_mean(grn_g, k_gr))
    put("look_gr_std", _fwd_std(grn_g, k_gr))
    k_dip = max(1, int(round(cfg["dip_trail_ft"] / step)))
    dip = np.empty(L)
    for i in range(L):
        j = max(0, i - k_dip)
        dm = md_g[i] - md_g[j]
        dip[i] = (base_g[i] - base_g[j]) / dm if dm > 0 else 0.0
    put("node_dip", dip * 20.0)
    put("ir", np.full(L, ir * 50.0))

    # ---- positional ----
    put("md_since", (md_g - last_md) / ms)
    put("dist_end", (md_g[-1] - md_g) / ms)
    put("idx_norm", np.arange(L) / max(L - 1, 1))
    put("total_len", np.full(L, (md_g[-1] - md_g[0]) / ms))

    # ---- alternative-family disagreement ----
    for src, nm in (("sp_blate", "fam_sp_blate"), ("beam_mean", "fam_beam_mean"),
                    ("poly1_t500", "fam_poly1_t500"), ("flat_hold", "fam_flat")):
        put(nm, (lin(cands[ni[src]]) - base_g) / ts)

    np.clip(X, -cl, cl, out=X)
    return X.astype(np.float32), md_g, base_g


# ---------------------------------------------------------------- caching driver
def load_dump(wid):
    with np.load(DUMP_DIR / f"{wid}.npz", allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def raw_gap_gr(wid):
    hw = pd.read_csv(TRAIN_DIR / f"{wid}__horizontal_well.csv")
    ev = hw[hw["TVT_input"].isna()]
    return hw["GR"].to_numpy(float)[ev.index]


def cache_one(wid):
    out = FEAT_DIR / f"{wid}.npz"
    if out.exists():
        return "skip"
    d = load_dump(wid)
    tvt_true = d.pop("tvt_true").astype(np.float64)      # label: never enters build_features
    X, md_g, base_g = build_features(d, raw_gap_gr(wid))
    md = d["md_ev"].astype(np.float64)
    y_g = np.interp(md_g, md, tvt_true) - base_g
    base_nat = d["cands"][[str(n) for n in d["names"]].index("pfens_s5")].astype(np.float64)
    np.savez_compressed(
        out, X=X, md_g=md_g.astype(np.float32), base_g=base_g.astype(np.float32),
        y_g=y_g.astype(np.float32),
        nat_md=md.astype(np.float32), nat_base=base_nat.astype(np.float32),
        nat_tvt=tvt_true.astype(np.float32))
    return "ok"


def canary(wids):
    """Truth canary: perturbing tvt_true must not change the features (bit-identical)."""
    rng = np.random.default_rng(CONFIG["seed"])
    for wid in wids:
        d = load_dump(wid)
        d.pop("tvt_true")
        g = raw_gap_gr(wid)
        X0, _, _ = build_features(dict(d), g)
        d2 = dict(d)
        d2["tvt_true"] = rng.normal(size=10)  # present but must be ignored
        X1, _, _ = build_features(d2, g)
        assert np.array_equal(X0, X1), f"canary FAIL {wid}"
    print(f"[feat] truth canary PASS on {len(wids)} wells")


def main():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    wids = sorted(p.stem for p in DUMP_DIR.glob("*.npz"))
    print(f"[feat] wells={len(wids)} n_ch={N_CH}")
    canary(wids[:3])
    res = Parallel(n_jobs=CONFIG["n_jobs"], verbose=1)(delayed(cache_one)(w) for w in wids)
    n_ok = sum(r == "ok" for r in res); n_skip = sum(r == "skip" for r in res)
    print(f"[feat] cached ok={n_ok} skip={n_skip} -> {FEAT_DIR}")


if __name__ == "__main__":
    main()
