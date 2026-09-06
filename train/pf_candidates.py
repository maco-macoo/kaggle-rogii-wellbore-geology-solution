"""Candidate-path generator library for the training-side PF dumps.

Per well (eval-zone arrays):
  A: 128-seed PF per-seed paths + likelihood-weighted ensemble means
  C: 7-config beam-search paths + their mean
  E: trivial prefix polyfit extrapolations + flat hold

The PF constants and code match the submission kernel's p128_* filter exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

# ---- constants (identical to the kernel) ------------------------------------
P128_N_PARTICLES = 500; P128_LIK_SCALE = 5.0; P128_INIT_SPREAD = 2.0
P128_MOM = 0.998; P128_VN = 0.002; P128_PN = 0.005; P128_RP = 0.1; P128_RR = 0.001; P128_RESAMP = 0.5
P128_GR_SIG_MIN = 10.0; P128_GR_SIG_MAX = 60.0
BEAMS = [(10, 20.0, 144.0, 2, "cons"), (10, 8.0, 64.0, 2, "loose"), (8, 35.0, 220.0, 1, "vcons"),
         (10, 14.0, 90.0, 5, "sm5"), (20, 4.0, 36.0, 3, "vloose"), (12, 12.0, 100.0, 3, "mid"),
         (15, 25.0, 180.0, 2, "stiff")]
LIK_SCALES = (3.0, 5.0, 8.0, 12.0)


# ---- family A: 128-seed PF per-seed paths -----------------------------------
@njit(cache=True)
def p128_interp(xs, xp, fp):
    out = np.empty(len(xs))
    for k in range(len(xs)):
        x = xs[k]
        if x <= xp[0]: out[k] = fp[0]
        elif x >= xp[-1]: out[k] = fp[-1]
        else:
            i = np.searchsorted(xp, x) - 1; t = (x - xp[i]) / (xp[i + 1] - xp[i]); out[k] = fp[i] + t * (fp[i + 1] - fp[i])
    return out


@njit(cache=True)
def p128_core(md_v, z_v, gr_v, tw_tvt, tw_gr, n_particles, seed, gs, last_tvt, last_z, last_md, ir,
              mom, vn, pn, rp, rr, resamp, init_spread, tmin, tmax):
    np.random.seed(seed)
    ls = last_tvt + last_z; pos = ls + init_spread * np.random.randn(n_particles)
    rate = ir + 0.01 * np.random.randn(n_particles); w = np.ones(n_particles) / n_particles
    res = np.empty(len(md_v)); log_lik = 0.0; prev_md = last_md
    for i in range(len(md_v)):
        dm = max(md_v[i] - prev_md, 1.0); rate = mom * rate + vn * np.random.randn(n_particles)
        pos = pos + rate * dm + pn * np.random.randn(n_particles)
        tvt_p = np.clip(pos - z_v[i], tmin - 100.0, tmax + 100.0); pos = tvt_p + z_v[i]
        if not np.isnan(gr_v[i]):
            eg = p128_interp(tvt_p, tw_tvt, tw_gr); d = (gr_v[i] - eg) / gs
            lk = np.exp(-0.5 * np.minimum(d ** 2, 600.0)); lk = np.maximum(lk, 1e-300)
            avg = np.dot(w, lk); log_lik += np.log(max(avg, 1e-300)); w = w * lk; ws = w.sum()
            w = w / ws if ws > 0.0 else np.ones(n_particles) / n_particles
        n_eff = 1.0 / np.dot(w, w)
        if n_eff < resamp * n_particles:
            cum = np.cumsum(w); u0 = np.random.uniform(0.0, 1.0 / n_particles)
            new_pos = np.empty(n_particles); new_rate = np.empty(n_particles)
            for j in range(n_particles):
                u = u0 + j / n_particles; idx = min(np.searchsorted(cum, u), n_particles - 1)
                new_pos[j] = pos[idx] + rp * np.random.randn(); new_rate[j] = rate[idx] + rr * np.random.randn()
            pos = new_pos; rate = new_rate; w = np.ones(n_particles) / n_particles
        res[i] = np.dot(w, pos) - z_v[i]; prev_md = md_v[i]
    return res, log_lik


def p128_gr_sigma(hw, tw_tvt, tw_gr):
    kn = hw[hw["TVT_input"].notna()]
    if len(kn) < 20: return 30.0
    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    return float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), P128_GR_SIG_MIN, P128_GR_SIG_MAX))


def p128_seed_paths(hw, tw_tvt, tw_gr, n_seeds, n_particles=P128_N_PARTICLES):
    """All per-seed eval-zone paths + log-likelihoods (kernel p128_run_pf, seed loop lifted out)."""
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    last_tvt = float(kn.iloc[-1]["TVT_input"]); last_z = float(kn.iloc[-1]["Z"]); last_md = float(kn.iloc[-1]["MD"])
    gs = p128_gr_sigma(hw, tw_tvt, tw_gr)
    tail = kn.tail(30); dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]
    tmin, tmax = float(tw_tvt.min()), float(tw_tvt.max())
    md_v = ev["MD"].values.astype(float); z_v = ev["Z"].values.astype(float)
    paths = np.empty((n_seeds, len(ev)), np.float32); liks = np.empty(n_seeds)
    for s in range(n_seeds):
        res, ll = p128_core(md_v, z_v, gr_v, tw_tvt, tw_gr, n_particles, s, gs,
                            last_tvt, last_z, last_md, ir, P128_MOM, P128_VN, P128_PN,
                            P128_RP, P128_RR, P128_RESAMP, P128_INIT_SPREAD, tmin, tmax)
        paths[s] = res; liks[s] = ll
    return paths, liks


def pf_derived(paths, liks):
    out = {}
    for sc in LIK_SCALES:
        w = np.exp((liks - liks.max()) / sc); w /= w.sum()
        out[f"pfens_s{sc:g}"] = (w[:, None] * paths).sum(0).astype(np.float32)
    order = np.argsort(liks)[::-1]
    out["pf_top1lik"] = paths[order[0]].copy()
    out["pf_top8lik"] = paths[order[:8]].mean(0).astype(np.float32)
    return out


def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


# ---- family C: beam-search dynamic programming ------------------------------
@njit(cache=True)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt: continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX: cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]; cC[i], cC[mi] = cC[mi], cC[i]; cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n - 1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path


def _nn(arr, v):
    i = int(np.searchsorted(arr, v, 'left'))
    if i >= len(arr): return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v): return i - 1
    return i


def _smooth(vals, fb, r):
    s = pd.Series(vals, dtype='float32').interpolate(limit_direction='both').fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)


def beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _nn(tw_tvt, start_tvt); sgr = _smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    return tw_tvt[_beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))].astype(np.float32)


def beam_family(hw, tw_tvt, tw_gr):
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    last_tvt = float(kn["TVT_input"].iloc[-1])
    gr_ev = hw["GR"].interpolate(limit_direction="both").values.astype(float)[ev.index]
    out = {}
    paths = []
    for bs, mc, es, r, name in BEAMS:
        p = beam_search(gr_ev, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
        out[f"beam_{name}"] = p; paths.append(p)
    stack = np.stack(paths, 0)
    out["beam_mean"] = stack.mean(0).astype(np.float32)
    out["beam_med"] = np.median(stack, 0).astype(np.float32)
    return out


# ---- family E: trivial prefix-tail extrapolations ---------------------------
def trivial_family(hw):
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    ev_md = ev["MD"].values.astype(float)
    last_tvt = float(kn["TVT_input"].iloc[-1])
    out = {"flat_hold": np.full(len(ev), last_tvt, np.float32)}
    for tail_n in (200, 500):
        t = kn.tail(tail_n)
        x = t["MD"].values.astype(float); y = t["TVT_input"].values.astype(float)
        for deg in (1, 2):
            if len(t) < 10 * deg or np.std(x) < 1e-6:
                out[f"poly{deg}_t{tail_n}"] = np.full(len(ev), last_tvt, np.float32)
                continue
            c = np.polyfit(x, y, deg)
            out[f"poly{deg}_t{tail_n}"] = np.polyval(c, ev_md).astype(np.float32)
    return out
