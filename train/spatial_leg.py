"""Spatial-surface leg: transfer the geology from neighboring training wells.

Structural identity: TVT = ANCC(X, Y) - Z + b_well
  ANCC   = the ANCC stratigraphic surface, reconstructed by IDW over the
           row-level dense point clouds of neighboring training wells
  b_well = per-well offset fixed from the known prefix as
           median(TVT_input + Z - ANCC_hat)  (leak-free)

Self-contained (numpy/scipy/pandas only); all settings live in CONFIG.
Used both embedded in the submission kernel and imported by train/pf_dump.py
(the LOO spatial feature fed to the GRU during training).
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


CONFIG = dict(
    SPW=60,               # samples per well for the point cloud (0 = all rows)
    K=20,                 # IDW neighbor count
    POWER=2.0,            # IDW distance power
    VARIANT="idw_trend",  # selected variant (global plane trend + IDW residual)
    ANISO_LAMBDA=1.0,     # anisotropy compression (idw_aniso only, unused here)
    ANISO_MODE="pca",     # "pca" | "heading"
    B_CAL="late",         # b calibration: "full" | "late"
    # -- guard thresholds (calibrated on the by-well CV curves) ---------------
    # NN_DIST_MAX: nearest-neighbor distance (scaled units) past which the
    #   surface reconstruction degrades sharply -> skip such wells.
    # PREFIX_RESID_MAX: self-validation RMSE of the reconstruction against the
    #   known prefix; wells above it are not trustworthy for this leg.
    NN_DIST_MAX=0.010,    # nearest-neighbor distance threshold (scaled)
    PREFIX_MIN=30,        # minimum known-prefix rows (below -> skip)
    PREFIX_RESID_MAX=20.0,  # prefix RMSE threshold (ft)
)


# ---------------------------------------------------------------- DenseCloud
class DenseCloud:
    """Row-level (X, Y, ANCC) point cloud over the training wells.
    Thinned per well by linspace (spw); coordinates scaled by their std.
    """

    def __init__(self):
        self.xy: np.ndarray = np.zeros((0, 2), np.float32)
        self.ancc: np.ndarray = np.zeros(0, np.float32)
        self.wids: np.ndarray = np.array([], dtype=object)
        self.scale: np.ndarray = np.ones(2, np.float64)
        self.heading: dict[str, float] = {}  # wid -> mean heading (rad)

    @classmethod
    def build(cls, train_dir: Path, spw: int = 60) -> "DenseCloud":
        """Build the point cloud from every __horizontal_well.csv in train_dir."""
        c = cls()
        xs, ys, anccs, wids = [], [], [], []
        headings: dict[str, float] = {}
        for p in sorted(train_dir.glob("*__horizontal_well.csv")):
            wid = p.stem.replace("__horizontal_well", "")
            try:
                df = pd.read_csv(p, usecols=["X", "Y", "ANCC"]).dropna()
            except Exception:
                continue
            if len(df) < 2:
                continue
            n = len(df)
            if spw > 0 and n > spw:
                ix = np.linspace(0, n - 1, spw, dtype=int)
            else:
                ix = np.arange(n)
            s = df.iloc[ix]
            xs.append(s["X"].values.astype(np.float32))
            ys.append(s["Y"].values.astype(np.float32))
            anccs.append(s["ANCC"].values.astype(np.float32))
            wids.extend([wid] * len(ix))
            # per-well mean heading (mean of 2*theta)
            dx = np.diff(df["X"].values)
            dy = np.diff(df["Y"].values)
            angles = np.arctan2(dy, dx)
            headings[wid] = float(np.angle(np.mean(np.exp(2j * angles)))) / 2
        c.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)]).astype(np.float64)
        c.ancc = np.concatenate(anccs).astype(np.float32)
        c.wids = np.array(wids, dtype=object)
        raw_std = c.xy.std(0)
        c.scale = np.where(raw_std < 1e-3, 1.0, raw_std)
        c.heading = headings
        return c

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        head_wids = np.array(list(self.heading.keys()), dtype=object)
        head_vals = np.array(list(self.heading.values()), dtype=np.float32)
        np.savez_compressed(
            str(path),
            xy=self.xy.astype(np.float32),
            ancc=self.ancc,
            wids=self.wids,
            scale=self.scale,
            head_wids=head_wids,
            head_vals=head_vals,
        )

    @classmethod
    def load(cls, path: Path) -> "DenseCloud":
        d = np.load(str(path), allow_pickle=True)
        c = cls()
        c.xy = d["xy"].astype(np.float64)
        c.ancc = d["ancc"].astype(np.float32)
        c.wids = d["wids"]
        c.scale = d["scale"].astype(np.float64)
        c.heading = dict(zip(d["head_wids"].tolist(), d["head_vals"].tolist()))
        return c


# ---------------------------------------------------------------- predictor classes
class _IdwPredictor:
    """Plain IDW: isotropic scaled distance; the self well's points are excluded."""

    def __init__(self, cloud: DenseCloud, k: int, power: float):
        self.cloud = cloud
        self.k = k
        self.power = power
        xy_sc = cloud.xy / cloud.scale
        self.tree = cKDTree(xy_sc)

    def predict(self, self_wid: str, xy_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        xy_q: (N,2) raw coordinates
        returns: (ancc_hat [N], nn_dist_scaled [N])
        """
        xy_q = np.atleast_2d(xy_q).astype(np.float64)
        q_sc = xy_q / self.cloud.scale
        nfetch = min(self.k + 500, len(self.cloud.ancc))
        dist, idx = self.tree.query(q_sc, k=nfetch, workers=-1)
        # drop the query well's own points
        mask_self = self.cloud.wids[idx] == self_wid
        dist = np.where(mask_self, np.inf, dist)
        k_use = min(self.k, nfetch)
        ord_ = np.argpartition(dist, min(k_use - 1, nfetch - 1), axis=1)[:, :k_use]
        dk = np.take_along_axis(dist, ord_, axis=1)
        ik = np.take_along_axis(idx, ord_, axis=1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk ** self.power + 1e-9), 0.0)
        an = self.cloud.ancc[ik].astype(np.float64)
        sw = w.sum(1)
        safe = np.where(sw < 1e-12, 1.0, sw)
        ancc_hat = (w * an).sum(1) / safe
        ancc_hat = np.where(sw < 1e-12, float(self.cloud.ancc.mean()), ancc_hat)
        nn_dist = np.where(vk, dk, np.inf).min(1)
        return ancc_hat.astype(np.float32), nn_dist.astype(np.float32)


class _IdwAnisoPredictor:
    """Anisotropic IDW: rotate to the principal heading and compress the
    orthogonal axis by lambda."""

    def __init__(self, cloud: DenseCloud, k: int, power: float,
                 lam: float = 1.0, mode: str = "pca"):
        self.cloud = cloud
        self.k = k
        self.power = power
        # transform matrix
        xy_sc = cloud.xy / cloud.scale
        if mode == "pca":
            mu = xy_sc.mean(0)
            c = xy_sc - mu
            cov = (c.T @ c) / max(len(c) - 1, 1)
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            self.R = evecs[:, order]  # (2,2)
            self.mu = mu
        else:  # heading
            angles = np.array(list(cloud.heading.values()))
            mean_angle = float(np.angle(np.mean(np.exp(2j * angles)))) / 2
            self.R = np.array([[np.cos(mean_angle), -np.sin(mean_angle)],
                                [np.sin(mean_angle),  np.cos(mean_angle)]])
            self.mu = xy_sc.mean(0)
        # compression: principal axis kept, orthogonal axis scaled by lam
        self.compress = np.array([1.0, lam])
        txy = ((xy_sc - self.mu) @ self.R) * self.compress
        self.tree = cKDTree(txy)
        self.txy = txy

    def _transform(self, xy_q: np.ndarray) -> np.ndarray:
        q_sc = xy_q / self.cloud.scale
        return ((q_sc - self.mu) @ self.R) * self.compress

    def predict(self, self_wid: str, xy_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy_q = np.atleast_2d(xy_q).astype(np.float64)
        tq = self._transform(xy_q)
        nfetch = min(self.k + 500, len(self.cloud.ancc))
        dist, idx = self.tree.query(tq, k=nfetch, workers=-1)
        mask_self = self.cloud.wids[idx] == self_wid
        dist = np.where(mask_self, np.inf, dist)
        k_use = min(self.k, nfetch)
        ord_ = np.argpartition(dist, min(k_use - 1, nfetch - 1), axis=1)[:, :k_use]
        dk = np.take_along_axis(dist, ord_, axis=1)
        ik = np.take_along_axis(idx, ord_, axis=1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk ** self.power + 1e-9), 0.0)
        an = self.cloud.ancc[ik].astype(np.float64)
        sw = w.sum(1)
        safe = np.where(sw < 1e-12, 1.0, sw)
        ancc_hat = (w * an).sum(1) / safe
        ancc_hat = np.where(sw < 1e-12, float(self.cloud.ancc.mean()), ancc_hat)
        nn_dist = np.where(vk, dk, np.inf).min(1)
        return ancc_hat.astype(np.float32), nn_dist.astype(np.float32)


class _IdwTrendPredictor:
    """Subtract a global plane trend, then IDW the residuals."""

    def __init__(self, cloud: DenseCloud, k: int, power: float):
        self.cloud = cloud
        self.k = k
        self.power = power
        xy_sc = cloud.xy / cloud.scale
        self.xy_sc = xy_sc
        # global plane fit (all points)
        ok = np.isfinite(cloud.ancc)
        mu = xy_sc[ok].mean(0)
        sc = np.where(xy_sc[ok].std(0) < 1e-9, 1.0, xy_sc[ok].std(0))
        A = np.column_stack([np.ones(ok.sum()), (xy_sc[ok] - mu) / sc])
        beta, *_ = np.linalg.lstsq(A, cloud.ancc[ok].astype(np.float64), rcond=None)
        self.mu = mu
        self.sc = sc
        self.beta = beta
        self.resid = cloud.ancc.astype(np.float64) - self._trend(xy_sc)
        self.tree = cKDTree(xy_sc)

    def _trend(self, xy_sc: np.ndarray) -> np.ndarray:
        A = np.column_stack([np.ones(len(xy_sc)), (xy_sc - self.mu) / self.sc])
        return A @ self.beta

    def predict(self, self_wid: str, xy_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy_q = np.atleast_2d(xy_q).astype(np.float64)
        q_sc = xy_q / self.cloud.scale
        trend_q = self._trend(q_sc)
        nfetch = min(self.k + 500, len(self.cloud.ancc))
        dist, idx = self.tree.query(q_sc, k=nfetch, workers=-1)
        mask_self = self.cloud.wids[idx] == self_wid
        dist = np.where(mask_self, np.inf, dist)
        k_use = min(self.k, nfetch)
        ord_ = np.argpartition(dist, min(k_use - 1, nfetch - 1), axis=1)[:, :k_use]
        dk = np.take_along_axis(dist, ord_, axis=1)
        ik = np.take_along_axis(idx, ord_, axis=1)
        vk = np.isfinite(dk)
        w = np.where(vk, 1.0 / (dk ** self.power + 1e-9), 0.0)
        rn = np.where(vk, self.resid[ik], 0.0)
        sw = w.sum(1)
        safe = np.where(sw < 1e-12, 1.0, sw)
        r_hat = (w * rn).sum(1) / safe
        r_hat = np.where(sw < 1e-12, 0.0, r_hat)
        ancc_hat = trend_q + r_hat
        nn_dist = np.where(vk, dk, np.inf).min(1)
        return ancc_hat.astype(np.float32), nn_dist.astype(np.float32)


class _WlsLocalPredictor:
    """Local IRLS (Huber) plane: weighted local plane fit on the k neighbors."""

    def __init__(self, cloud: DenseCloud, k: int, power: float):
        self.cloud = cloud
        self.k = k
        self.power = power
        xy_sc = cloud.xy / cloud.scale
        self.xy_sc = xy_sc
        self.tree = cKDTree(xy_sc)

    def predict(self, self_wid: str, xy_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy_q = np.atleast_2d(xy_q).astype(np.float64)
        q_sc = xy_q / self.cloud.scale
        nfetch = min(self.k + 500, len(self.cloud.ancc))
        dist, idx = self.tree.query(q_sc, k=nfetch, workers=-1)
        mask_self = self.cloud.wids[idx] == self_wid
        dist = np.where(mask_self, np.inf, dist)
        k_use = min(self.k, nfetch)
        ord_ = np.argpartition(dist, min(k_use - 1, nfetch - 1), axis=1)[:, :k_use]
        dk = np.take_along_axis(dist, ord_, axis=1)
        ik = np.take_along_axis(idx, ord_, axis=1)
        ancc_hat = np.zeros(len(xy_q), np.float64)
        nn_dist = np.zeros(len(xy_q), np.float32)
        for i in range(len(xy_q)):
            vk_i = np.isfinite(dk[i])
            nn_dist[i] = float(dk[i][vk_i].min()) if vk_i.any() else np.inf
            if not vk_i.any():
                ancc_hat[i] = float(self.cloud.ancc.mean())
                continue
            nb_xy = self.xy_sc[ik[i][vk_i]]  # (K,2)
            nb_an = self.cloud.ancc[ik[i][vk_i]].astype(np.float64)
            nb_d = dk[i][vk_i]
            qc = q_sc[i]
            # center on the query point
            dx = nb_xy[:, 0] - qc[0]
            dy = nb_xy[:, 1] - qc[1]
            A = np.column_stack([np.ones(len(dx)), dx, dy])
            w_init = 1.0 / (nb_d ** self.power + 1e-9)
            # IRLS (Huber, 5 iter)
            w = w_init.copy()
            for _ in range(5):
                W = np.diag(w)
                try:
                    beta = np.linalg.solve(A.T @ W @ A + 1e-6 * np.eye(3), A.T @ W @ nb_an)
                except np.linalg.LinAlgError:
                    beta = np.array([float(np.nanmean(nb_an)), 0.0, 0.0])
                    break
                resid = nb_an - A @ beta
                sigma = max(np.median(np.abs(resid)) * 1.4826, 1e-6)
                r_sc = resid / sigma
                delta = 1.345
                huber_w = np.where(np.abs(r_sc) <= delta, 1.0, delta / np.abs(r_sc))
                w = w_init * huber_w
            ancc_hat[i] = beta[0]  # b0 = intercept at the centered origin = value at the query point
        return ancc_hat.astype(np.float32), nn_dist


def make_predictor(cloud: DenseCloud, variant: str = "idw", **kw):
    """Instantiate the predictor for the requested variant."""
    k = int(kw.get("k", CONFIG["K"]))
    power = float(kw.get("power", CONFIG["POWER"]))
    if variant == "idw":
        return _IdwPredictor(cloud, k, power)
    elif variant == "idw_aniso":
        lam = float(kw.get("lam", CONFIG["ANISO_LAMBDA"]))
        mode = str(kw.get("mode", CONFIG["ANISO_MODE"]))
        return _IdwAnisoPredictor(cloud, k, power, lam=lam, mode=mode)
    elif variant == "idw_trend":
        return _IdwTrendPredictor(cloud, k, power)
    elif variant == "wls_local":
        return _WlsLocalPredictor(cloud, k, power)
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------- b calibration
def _robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Drift slope of b vs measured depth."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) < 1e-6:
        return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])


def spatial_tvt_for_well(
    hw_df: pd.DataFrame,
    predictor,
    self_wid: str,
) -> dict:
    """
    One well: dense ANCC IDW -> three b calibrations -> three TVT_sp series
    plus diagnostics.

    Returns a dict with:
      kn_idx / ev_idx        prefix and eval row indices
      ancc_hat_all           ANCC_hat on all rows (raw coords)
      nn_dist_all            nearest-neighbor distance on all rows (scaled)
      b_full / b_late / b_slope / b_PS / md_PS
      tvt_bfull / tvt_blate / tvt_drift   eval-row TVT_sp series
      prefix_resid_rmse      prefix self-validation RMSE (TVT_sp vs TVT_input)
      n_kn / n_ev / nn_dist_ev / md_since
    """
    kn = hw_df[hw_df["TVT_input"].notna()]
    ev = hw_df[hw_df["TVT_input"].isna()]
    if len(ev) == 0 or len(kn) == 0:
        return {}
    kn_idx = kn.index.to_numpy()
    ev_idx = ev.index.to_numpy()
    # predict all rows
    all_xy = hw_df[["X", "Y"]].to_numpy(np.float64)
    ancc_hat_all, nn_dist_all = predictor.predict(self_wid, all_xy)

    # calibrate b on the prefix (b = TVT_input + Z - ANCC_hat)
    kn_z = kn["Z"].to_numpy(np.float64)
    kn_tvti = kn["TVT_input"].to_numpy(np.float64)
    ancc_hat_kn = ancc_hat_all[kn_idx - hw_df.index[0]]
    b_vec = kn_tvti + kn_z - ancc_hat_kn
    n_kn = len(kn)
    b_full = float(np.median(b_vec))
    # b_late: last 50 prefix rows
    b_late = float(np.median(b_vec[max(0, n_kn - 50):])) if n_kn >= 5 else b_full
    # drift calibration: extrapolate the MD drift of b from the last 200 rows
    md_kn = kn["MD"].to_numpy(np.float64)
    md_ev = ev["MD"].to_numpy(np.float64)
    md_PS = float(md_kn[-1])
    tail_sl = slice(max(0, n_kn - 200), n_kn)
    b_slope = _robust_slope(md_kn[tail_sl] - md_PS, b_vec[tail_sl])
    md_since = md_ev - md_PS
    b_PS = b_vec[-1] if len(b_vec) > 0 else b_full

    # prefix self-validation
    prefix_resid_rmse = float(np.sqrt(np.mean((kn_tvti - (ancc_hat_kn - kn_z + b_full)) ** 2)))

    # eval row indices (assumes a reset index)
    ri = hw_df.index[0]
    ev_rel = ev_idx - ri
    ev_z = ev["Z"].to_numpy(np.float64)
    ancc_hat_ev = ancc_hat_all[ev_rel]

    tvt_bfull = ancc_hat_ev - ev_z + b_full
    tvt_blate = ancc_hat_ev - ev_z + b_late
    tvt_drift = ancc_hat_ev - ev_z + (b_PS + b_slope * md_since)

    return {
        "kn_idx": kn_idx,
        "ev_idx": ev_idx,
        "ancc_hat_all": ancc_hat_all,
        "nn_dist_all": nn_dist_all,
        "b_full": b_full,
        "b_late": b_late,
        "b_slope": b_slope,
        "b_PS": b_PS,
        "md_PS": md_PS,
        "tvt_bfull": tvt_bfull,
        "tvt_blate": tvt_blate,
        "tvt_drift": tvt_drift,
        "prefix_resid_rmse": prefix_resid_rmse,
        "n_kn": n_kn,
        "n_ev": len(ev),
        "nn_dist_ev": nn_dist_all[ev_rel],
        "md_since": md_since,
    }


# ---------------------------------------------------------------- entry point
def run_spatial_leg(
    test_dir: Path,
    train_dir: Path,
    cfg: dict | None = None,
    guards: bool = True,
) -> tuple[dict, dict]:
    """
    Entry point: produce the spatial TVT_sp for every well in test_dir.

    Kernel-embeddable: no Path(__file__) references. The DenseCloud is always
    rebuilt from train_dir (no cache; takes ~1-2 minutes).

    Args:
      test_dir:  directory of wells to predict
      train_dir: directory the DenseCloud is built from
      cfg:       CONFIG dict (None -> the frozen defaults above)
      guards:    True enables NN_DIST_MAX / PREFIX_RESID_MAX
                 (PREFIX_MIN is always enforced)

    returns:
      id_to_tvt:   {row_id: tvt_blate}
      id_to_valid: {row_id: True/False}  (guard verdict)
    """
    import time as _time
    t0 = _time.time()
    if cfg is None:
        cfg = CONFIG
    b_cal = cfg.get("B_CAL", "late")
    print(f"[spatial_leg] Building DenseCloud from {train_dir} spw={cfg['SPW']}...", flush=True)
    cloud = DenseCloud.build(Path(train_dir), spw=cfg["SPW"])
    print(f"[spatial_leg] Cloud built: {len(cloud.ancc)} pts ({_time.time()-t0:.1f}s)", flush=True)
    pred = make_predictor(cloud, cfg["VARIANT"],
                          k=cfg["K"], power=cfg["POWER"],
                          lam=cfg.get("ANISO_LAMBDA", 1.0),
                          mode=cfg.get("ANISO_MODE", "pca"))
    id_to_tvt: dict[str, float] = {}
    id_to_valid: dict[str, bool] = {}
    n_applied = 0
    for hw_path in sorted(Path(test_dir).glob("*__horizontal_well.csv")):
        wid = hw_path.stem.replace("__horizontal_well", "")
        try:
            hw = pd.read_csv(hw_path)
        except Exception:
            continue
        hw = hw.reset_index(drop=True)
        kn = hw[hw["TVT_input"].notna()]
        ev = hw[hw["TVT_input"].isna()]
        if len(ev) == 0:
            continue
        n_kn = len(kn)
        # PREFIX_MIN is checked regardless of the guards flag
        valid = n_kn >= cfg["PREFIX_MIN"]
        res = spatial_tvt_for_well(hw, pred, wid)
        if not res:
            for i in ev.index:
                id_to_tvt[f"{wid}_{i}"] = float(kn["TVT_input"].iloc[-1]) if n_kn > 0 else 0.0
                id_to_valid[f"{wid}_{i}"] = False
            continue
        # NN_DIST_MAX / PREFIX_RESID_MAX apply only with guards=True
        nn_med = float(np.median(res["nn_dist_ev"])) if len(res["nn_dist_ev"]) else np.inf
        nn_max = float(res["nn_dist_ev"].max()) if len(res["nn_dist_ev"]) else np.inf
        guard_nn = True
        guard_pr = True
        if guards:
            if cfg.get("NN_DIST_MAX") is not None:
                guard_nn = nn_max <= cfg["NN_DIST_MAX"]
                valid = valid and guard_nn
            if cfg.get("PREFIX_RESID_MAX") is not None:
                guard_pr = res["prefix_resid_rmse"] <= cfg["PREFIX_RESID_MAX"]
                valid = valid and guard_pr
        # choose the b calibration
        tvt_sp = res["tvt_blate"] if b_cal == "late" else res["tvt_bfull"]
        # per-well audit log
        guard_rate = float((~np.isfinite(res["nn_dist_ev"]) | (res["nn_dist_ev"] > cfg.get("NN_DIST_MAX", np.inf))).mean()) if guards and cfg.get("NN_DIST_MAX") else 0.0
        print(
            f"[spatial_leg] {wid}: n_kn={n_kn} n_ev={len(ev)} "
            f"b_{b_cal}={res['b_late' if b_cal=='late' else 'b_full']:.2f} "
            f"prefix_resid={res['prefix_resid_rmse']:.2f} "
            f"nn_dist_med={nn_med:.5f} nn_dist_max={nn_max:.5f} "
            f"guard_nn={guard_nn} guard_pr={guard_pr} valid={valid}",
            flush=True,
        )
        for j, i in enumerate(ev.index):
            id_to_tvt[f"{wid}_{i}"] = float(tvt_sp[j])
            id_to_valid[f"{wid}_{i}"] = valid
        if valid:
            n_applied += 1
    print(f"[spatial_leg] Done: {n_applied}/{len(list(Path(test_dir).glob('*__horizontal_well.csv')))} wells valid "
          f"({_time.time()-t0:.1f}s total)", flush=True)
    return id_to_tvt, id_to_valid


