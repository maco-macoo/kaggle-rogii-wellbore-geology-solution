"""GBDT-PP training: the boosted-tree stack behind the base prediction.

Recipe: 3x LightGBM + 2x CatBoost on the ~195-feature table
-> Ridge stack over physics features -> positive ridge blend
-> post-processing ensemble (top-8 softmax over an (alpha, tau, w_pf, sg) grid).
Validation: StratifiedGroupKFold(5) grouped by well.

Artifacts (train/artifacts/gbdt/): lgb0..2.txt, cb0..1.cbm, phys_ridge.pkl,
ridge_blend.pkl, pp.json, meta.json -- exactly what the submission kernel loads.
OOF: train/cache/oof_gbdt.npz.

Usage:
  python train/build_features.py     # first: the feature table
  python train/train_gbdt.py
"""
from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from scipy.signal import savgol_filter

CACHE = Path(__file__).resolve().parent / "cache"
ART = Path(__file__).parent / "artifacts" / "gbdt"; ART.mkdir(parents=True, exist_ok=True)

N_EVAL_THRESH = 4840.0
Z_SPAN_THRESH = (136.73, 185.51)
N_SPLITS = 5
BLEND_W = 0.50

# 3x LightGBM (CUDA): one large-num_leaves/low-regularization, two strongly regularized
LGB_CONFIGS = [
    dict(n_estimators=6000, learning_rate=0.030, num_leaves=255, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=0.05, min_child_samples=20, max_bin=255, random_state=123),
    dict(n_estimators=8000, learning_rate=0.012, num_leaves=64, subsample=0.6, subsample_freq=1,
         colsample_bytree=0.45, reg_lambda=80.0, reg_alpha=10.0, min_child_samples=40, max_bin=255, random_state=0),
    dict(n_estimators=8000, learning_rate=0.012, num_leaves=96, subsample=0.7, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=40.0, reg_alpha=5.0, min_child_samples=60, max_bin=255, random_state=29),
]
# 2x CatBoost (GPU)
CB_CONFIGS = [
    dict(iterations=8000, learning_rate=0.020, depth=7, l2_leaf_reg=2.0, random_seed=7),
    dict(iterations=8000, learning_rate=0.025, depth=8, l2_leaf_reg=8.0, random_seed=2024),
]
PHYS_DIRECT = ['pf_ancc_delta', 'pf_z_delta', 'hyb_d', 'beam_cons_d', 'beam_loose_d',
               'beam_vcons_d', 'beam_sm5_d', 'beam_vloose_d', 'beam_mid_d', 'beam_stiff_d',
               'beam_mean_d', 'beam_med_d', 'sc_cons_d', 'sc_ens_d', 'tvt_dense_d',
               'tvt_densew_d', 'tvt_dense50_d', 'form_mean_d', 'slp_b_d_all', 'slp_b_d_50']
PHYS_ALPHAS = (0.05, 0.2, 1.0, 5.0, 25.0, 100.0)


def assign_bins(df):
    g = df.groupby("well")
    n_eval = g["eval_len"].first(); z_span = g["z"].max() - g["z"].min()
    n_bin = (n_eval > N_EVAL_THRESH).astype(int)
    z_bin = pd.cut(z_span, [-np.inf, *Z_SPAN_THRESH, np.inf], labels=[0, 1, 2]).astype(int)
    return df["well"].map((n_bin + 2 * z_bin).astype(int))


def clean_num(v):
    return np.nan_to_num(np.asarray(v, np.float32), nan=0., posinf=0., neginf=0.).astype(np.float32)


def physics_cols(df):
    base = df['last_known_tvt'].to_numpy(np.float32); cand = {}
    direct = list(PHYS_DIRECT) + [c for c in df.columns if c.startswith(('tdbc', 'tdsc', 'tdpf'))]
    for c in dict.fromkeys(direct):
        if c in df.columns:
            v = clean_num(df[c].to_numpy(np.float32))
            if np.nanstd(v) > 1e-6: cand[c] = v
    abs_cols = ['pf_ancc', 'pf_z'] + [c for c in df.columns if c.startswith(('tvtF_', 'tvtFw_', 'tvtF50_'))]
    for c in dict.fromkeys(abs_cols):
        if c in df.columns:
            v = clean_num(df[c].to_numpy(np.float32) - base)
            if np.nanstd(v) > 1e-6: cand[f'{c}_delta'] = v
    return pd.DataFrame(cand).replace([np.inf, -np.inf], np.nan).fillna(0.).astype(np.float32)


def sg_smooth_groups(well, vals, sg_w=17, sg_p=3):
    out = vals.copy(); df = pd.DataFrame({"well": well, "v": vals})
    for _, g in df.groupby("well", sort=False):
        v = g["v"].values; n = len(v); wl = min(sg_w, n)
        if wl % 2 == 0: wl -= 1
        if wl >= sg_p + 2: out[g.index.values] = savgol_filter(v, wl, sg_p)
    return out


def apply_pp(md_since, model_delta, phys_delta, alpha, tau, w_pf):
    d = model_delta * (1 - w_pf) + phys_delta * w_pf
    if tau: d = d * (1. - np.exp(-np.maximum(md_since, 0.) / tau))
    return d * alpha


def fit_lgb(cfg, X, y, eval_set=None):
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    try:
        m = LGBMRegressor(device_type="cuda", n_jobs=-1, **cfg)
        cb = [log_evaluation(0)] + ([early_stopping(120, verbose=False)] if eval_set else [])
        m.fit(X, y, eval_set=eval_set, callbacks=cb)
        return m
    except Exception as e:
        print(f"    [LGB CUDA fail -> CPU] {str(e)[:80]}")
        m = LGBMRegressor(device_type="cpu", n_jobs=-1, **cfg)
        cb = [log_evaluation(0)] + ([early_stopping(120, verbose=False)] if eval_set else [])
        m.fit(X, y, eval_set=eval_set, callbacks=cb)
        return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(CACHE / "train_features.pkl"))
    args = ap.parse_args()
    from catboost import CatBoostRegressor

    t0 = time.time()
    df = pd.read_pickle(args.features)
    feat_cols = [c for c in df.columns if c not in ("well", "id", "target")]
    X = df[feat_cols].astype(np.float32); y = df["target"].astype(np.float32).values
    groups = df["well"].values; bins = assign_bins(df).values
    base = df["last_known_tvt"].values.astype(np.float32)
    ytrue = y + base
    ids = df["id"].values
    print(f"train: {df.shape}  feat={len(feat_cols)}  wells={df['well'].nunique()}  ({time.time()-t0:.0f}s)")

    folds = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42).split(X, bins, groups))
    fold_id = np.full(len(X), -1, np.int32)
    for fi, (_, va) in enumerate(folds): fold_id[va] = fi

    base_oof = {}
    for i, cfg in enumerate(LGB_CONFIGS):
        ts = time.time(); oof = np.zeros(len(X), np.float32)
        for tr, va in folds:
            m = fit_lgb(cfg, X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])])
            oof[va] = m.predict(X.iloc[va])
        base_oof[f"lgb{i}"] = oof
        mf = fit_lgb(cfg, X, y); mf.booster_.save_model(str(ART / f"lgb{i}.txt"))
        print(f"  LGB{i}: OOF RMSE={root_mean_squared_error(y, oof):.4f}  ({time.time()-ts:.0f}s)")

    for i, cfg in enumerate(CB_CONFIGS):
        ts = time.time(); oof = np.zeros(len(X), np.float32)
        for tr, va in folds:
            m = CatBoostRegressor(task_type="GPU", devices='0', verbose=0,
                                  early_stopping_rounds=200, **cfg)
            m.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]))
            oof[va] = m.predict(X.iloc[va])
        base_oof[f"cb{i}"] = oof
        mf = CatBoostRegressor(task_type="GPU", devices='0', verbose=0, **cfg).fit(X, y)
        mf.save_model(str(ART / f"cb{i}.cbm"))
        print(f"  CB{i}: OOF RMSE={root_mean_squared_error(y, oof):.4f}  ({time.time()-ts:.0f}s)")

    # physics Ridge stack
    P = physics_cols(df); pcols = sorted(P.columns)
    scores = sorted((root_mean_squared_error(y, P[c].values), c) for c in pcols)
    keep = [c for _, c in scores[:min(96, len(scores))]]
    Xp = P[keep].to_numpy(np.float32)
    phys_oof = {}; phys_full = {}
    for a in PHYS_ALPHAS:
        so = np.zeros(len(Xp), np.float32)
        for f in range(N_SPLITS):
            tr = fold_id != f; va = fold_id == f
            mu = Xp[tr].mean(0); sd = Xp[tr].std(0); sd[sd < 1e-6] = 1.
            r = Ridge(alpha=a, fit_intercept=True).fit((Xp[tr] - mu) / sd, y[tr])
            so[va] = r.predict((Xp[va] - mu) / sd).astype(np.float32)
        phys_oof[f"phys_a{a}"] = so
        mu = Xp.mean(0); sd = Xp.std(0); sd[sd < 1e-6] = 1.
        rf = Ridge(alpha=a, fit_intercept=True).fit((Xp - mu) / sd, y)
        phys_full[f"phys_a{a}"] = dict(coef=rf.coef_, intercept=rf.intercept_, mu=mu, sd=sd)
    with open(ART / "phys_ridge.pkl", "wb") as f:
        pickle.dump({"keep_cols": keep, "models": phys_full}, f)

    # ridge blend (positive, alpha=1.66)
    blend_keys = list(base_oof) + list(phys_oof)
    O = np.column_stack([{**base_oof, **phys_oof}[k] for k in blend_keys]).astype(np.float64)
    ridge_oof = np.zeros(len(O), np.float32)
    for f in range(N_SPLITS):
        tr = fold_id != f; va = fold_id == f
        rbi = Ridge(alpha=1.66, positive=True, fit_intercept=True, tol=5e-4, random_state=42).fit(O[tr], y[tr])
        ridge_oof[va] = rbi.predict(O[va]).astype(np.float32)
    rb = Ridge(alpha=1.66, positive=True, fit_intercept=True, tol=5e-4, random_state=42).fit(O, y)
    with open(ART / "ridge_blend.pkl", "wb") as f:
        pickle.dump({"keys": blend_keys, "coef": rb.coef_, "intercept": rb.intercept_}, f)
    print(f"  ridge-blend OOF RMSE={root_mean_squared_error(ytrue, base + ridge_oof):.4f}")

    # post-processing grid -> top-8 softmax
    pf_single = (df['pf_ancc'].values - base).astype(np.float32)
    md_since = df['md_since'].values.astype(np.float32); well = df['well'].values
    grid = [{'alpha': a, 'tau': t, 'w_pf': w, 'sg_w': s}
            for a in [0.98, 0.99, 1.0, 1.01] for t in [35, 50, 65, 85, 105, 130, 170]
            for w in [0.03, 0.05, 0.07, 0.09, 0.11, 0.13] for s in [13, 17, 21]]
    pps = []
    for p in grid:
        d = apply_pp(md_since, ridge_oof, pf_single, p['alpha'], p['tau'], p['w_pf'])
        pred = sg_smooth_groups(well, (base + d).astype(np.float32), int(p['sg_w']))
        pps.append((float(root_mean_squared_error(ytrue, pred)), p))
    pps.sort(key=lambda x: x[0]); top = pps[:8]
    w = np.array([np.exp(-max(s - top[0][0], 0.) / 0.01) for s, _ in top]); w /= w.sum()
    with open(ART / "pp.json", "w") as f:
        json.dump({"params": [p for _, p in top], "weights": w.tolist()}, f, indent=2)

    pp_oof = np.zeros(len(ytrue), np.float32)
    for wt, (_, p) in zip(w, top):
        d = apply_pp(md_since, ridge_oof, pf_single, p['alpha'], p['tau'], p['w_pf'])
        pp_oof += wt * sg_smooth_groups(well, (base + d).astype(np.float32), int(p['sg_w']))
    gbdt_pp_rmse = root_mean_squared_error(ytrue, pp_oof)
    print(f"  PP ensemble OOF RMSE={gbdt_pp_rmse:.4f}")

    with open(ART / "meta.json", "w") as f:
        json.dump({"feat_cols": feat_cols, "blend_w": BLEND_W,
                   "lgb_keys": [k for k in base_oof if k.startswith("lgb")],
                   "cb_keys": [k for k in base_oof if k.startswith("cb")],
                   "phys_alphas": list(PHYS_ALPHAS)}, f)

    # OOF dump
    np.savez(CACHE / "oof_gbdt.npz", id=ids, final_oof=pp_oof.astype(np.float32),
             ridge_oof=(base + ridge_oof).astype(np.float32), bin=bins,
             well=groups, ytrue=ytrue.astype(np.float32), last_known_tvt=base)
    print(f"\nGBDT-PP OOF={gbdt_pp_rmse:.4f} ft  saved: train/cache/oof_gbdt.npz  total {time.time()-t0:.0f}s")
    for p in sorted(ART.iterdir()): print(f"  {p.name}")


if __name__ == "__main__":
    main()
