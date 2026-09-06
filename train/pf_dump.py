"""Step 1 of GRU training: per-well PF dumps over the 773 training wells.

For every training well, run the 128-seed particle filter and dump everything
the GRU feature builder reads:
  pf000..pf127 seed paths + log-likelihoods, likelihood-weighted ensembles
  (pfens_s*), top1/top8 paths, the beam-search mean, trivial extrapolations,
  the leave-one-out spatial-surface path (sp_blate), the typewell grid,
  leak-free prefix scalars, and the truth (tvt_true; label use only).

Resume-capable: existing npz files are skipped.

  python train/pf_dump.py                # full 773 wells
  python train/pf_dump.py --sample 24    # smoke run
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import pf_candidates as C
from spatial_leg import CONFIG as SP_CONFIG, run_spatial_leg

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = (Path("/kaggle/input/rogii-wellbore-geology-prediction")
             if Path("/kaggle").exists() else ROOT / "input") / "train"
ART = Path(__file__).resolve().parent / "artifacts"
DUMP_DIR = ART / "dump"

CONFIG = dict(n_seeds=128, n_particles=500, batch=40, n_jobs=8, seed=42)


def prefix_scalars(hw, tw_tvt, tw_gr):
    """Leak-free per-well scalars for the feature builder (prefix rows only)."""
    kn = hw[hw["TVT_input"].notna()]
    last_tvt = float(kn["TVT_input"].iloc[-1]); last_md = float(kn["MD"].iloc[-1])
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gr_sigma = C.p128_gr_sigma(hw, tw_tvt, tw_gr)
    return dict(last_tvt=last_tvt, last_md=last_md, ir=ir, gr_sigma=gr_sigma)


def process_well(wid, sp_map, n_seeds):
    try:
        hw = pd.read_csv(TRAIN_DIR / f"{wid}__horizontal_well.csv")
        tw = pd.read_csv(TRAIN_DIR / f"{wid}__typewell.csv")
        kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0 or len(kn) == 0 or "TVT" not in hw.columns:
            return None, "no eval zone / no truth"
        tws = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
        tw_tvt = tws["TVT"].to_numpy(float); tw_gr = tws["GR"].to_numpy(float)
        if len(tw_tvt) < 10:
            return None, "typewell too short"

        cands = {}
        paths, liks = C.p128_seed_paths(hw, tw_tvt, tw_gr, n_seeds)
        for s in range(n_seeds):
            cands[f"pf{s:03d}"] = paths[s]
        cands.update(C.pf_derived(paths, liks))
        cands.update(C.beam_family(hw, tw_tvt, tw_gr))
        cands.update(C.trivial_family(hw))
        # leave-one-out spatial-surface path; rows the leg could not produce
        # fall back to the PF ensemble (channel value 0 after normalization)
        sp = np.array([sp_map.get(f"{wid}_{i}", np.nan) for i in ev.index], np.float64)
        pf5 = cands["pfens_s5"].astype(np.float64)
        cands["sp_blate"] = np.where(np.isfinite(sp), sp, pf5).astype(np.float32)

        names = list(cands)
        P = np.stack([cands[k] for k in names], 0).astype(np.float32)
        truth = hw["TVT"].to_numpy(float)[ev.index]

        sc = prefix_scalars(hw, tw_tvt, tw_gr)
        gg, gmin, gstep = C._grid(tw_tvt, tw_gr)
        gr_ev = (hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean())
                 .values.astype(np.float32)[ev.index])
        np.savez_compressed(
            DUMP_DIR / f"{wid}.npz",
            cands=P, names=np.array(names),
            tvt_true=truth.astype(np.float32),           # label use ONLY
            gr_ev=gr_ev,
            md_ev=ev["MD"].values.astype(np.float32),
            z_ev=ev["Z"].values.astype(np.float32),
            gg=gg.astype(np.float32), gmin=np.float64(gmin), gstep=np.float64(gstep),
            liks=liks.astype(np.float64),
            **{k: np.float64(v) for k, v in sc.items()})
        base = float(np.sqrt(np.mean((cands["pfens_s5"].astype(np.float64) - truth) ** 2)))
        return dict(wid=wid, n_ev=len(ev), pfens_s5_rmse=base), None
    except Exception as e:  # noqa: BLE001 -- one bad well must not kill the sweep
        return None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="0 = all wells")
    ap.add_argument("--n-seeds", type=int, default=CONFIG["n_seeds"])
    ap.add_argument("--n-jobs", type=int, default=CONFIG["n_jobs"])
    args = ap.parse_args()

    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    wids = sorted(p.stem.replace("__horizontal_well", "")
                  for p in TRAIN_DIR.glob("*__horizontal_well.csv"))
    if args.sample > 0:
        rng = np.random.default_rng(CONFIG["seed"])
        wids = sorted(rng.choice(wids, size=args.sample, replace=False))

    # leave-one-out spatial paths for all training wells in one pass
    # (predict wells = train wells; the predictor excludes each well's own points)
    print("[dump] building LOO spatial paths ...", flush=True)
    sp_map, _ = run_spatial_leg(TRAIN_DIR, TRAIN_DIR, SP_CONFIG, guards=False)

    todo = [w for w in wids if not (DUMP_DIR / f"{w}.npz").exists()]
    print(f"[dump] wells={len(wids)} todo={len(todo)}", flush=True)
    t0 = time.time(); n_new = 0
    with Parallel(n_jobs=args.n_jobs) as par:
        for bi in range(0, len(todo), CONFIG["batch"]):
            batch = todo[bi:bi + CONFIG["batch"]]
            res = par(delayed(process_well)(w, sp_map, args.n_seeds) for w in batch)
            for w, (row, err) in zip(batch, res):
                if err is not None:
                    print(f"[dump] SKIP {w}: {err}", flush=True)
                    continue
                n_new += 1
            el = time.time() - t0
            done = bi + len(batch)
            eta = el / done * (len(todo) - done)
            print(f"[dump] {done}/{len(todo)} | elapsed {el/60:.1f}m | ETA {eta/60:.1f}m",
                  flush=True)
    total = len(list(DUMP_DIR.glob("*.npz")))
    print(f"[dump] done: +{n_new} this pass, {total} dumped in {DUMP_DIR}")


if __name__ == "__main__":
    main()
