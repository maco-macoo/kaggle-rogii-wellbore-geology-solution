"""Per-fold training of the BiGRU tail refiner.

The deployed ensemble is 5 folds x 3 seeds = 15 models (crop off):

  for f in 0 1 2 3 4; do for s in 42 43 44; do
    python train/gru_train.py --fold $f --seed $s
  done; done

Resume: a per-epoch checkpoint (model/opt/epoch/best/RNG) is kept in
artifacts/gru/ckpt/<tag>.pt; --minutes N exits rc=3 when the budget is hit
(relaunch the same command to continue). On completion the fold-test OOF
residuals go to artifacts/gru/oof/<tag>.npz, the final weights to
artifacts/gru/weights/<tag>.pt (what the kernel loads), and the checkpoint
is deleted.

Leak notes: model/hyperparameters never see the fold-test wells; early stop
uses an inner well-grouped split of fold-train (GroupKFold(7) first split);
inner-val is scored as native-row pooled RMSE (the competition metric).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gru_dataset import FEAT_DIR, WellStore, batches, epoch_items  # noqa: E402
from gru_model import RefinerGRU  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts" / "gru"

N_FOLDS = 5

CONFIG = dict(
    lr=1e-3, wd=1e-4, warmup_ep=3, max_ep=120, patience=15, batch=16,
    huber_delta=10.0, hidden=64, drop=0.2, inner_k=7, grad_clip=1.0,
)


def pooled(rmses, n_evs):
    """Row-pooled RMSE == the competition metric."""
    r = np.asarray(rmses, np.float64)
    n = np.asarray(n_evs, np.float64)
    return float(np.sqrt(np.sum(n * r * r) / np.sum(n)))


def fold_map(wids):
    """{wid: fold} -- deterministic assignment over the sorted well ids."""
    wid_sorted = np.array(sorted(wids))
    out = {}
    for f, (_, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(
            wid_sorted, groups=wid_sorted)):
        for w in wid_sorted[te]:
            out[w] = f
    return out


def masked_huber(pred, y, mask, delta):
    e = (pred - y).abs()
    l = torch.where(e <= delta, 0.5 * e * e, delta * (e - 0.5 * delta))
    return (l * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def predict_wells(model, store, wids, device):
    """Full-tail residual predictions interpolated to native rows: {wid: r_hat_nat}."""
    model.eval()
    out = {}
    for w in wids:
        d = store.wells[w]
        X = torch.from_numpy(d["X"][None]).to(device)
        L = torch.tensor([d["X"].shape[1]])
        r_g = model(X, L)[0].cpu().numpy().astype(np.float64)
        out[w] = np.interp(d["nat_md"].astype(np.float64),
                           d["md_g"].astype(np.float64), r_g)
    return out


def pooled_rmse(store, preds):
    rs, ns = [], []
    for w, r in preds.items():
        d = store.wells[w]
        pred = d["nat_base"].astype(np.float64) + r
        err = pred - d["nat_tvt"].astype(np.float64)
        rs.append(float(np.sqrt(np.mean(err ** 2)))); ns.append(len(err))
    return pooled(rs, ns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crop", default="off", choices=["on", "off"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--minutes", type=float, default=0, help="0 = no time budget")
    ap.add_argument("--max-ep", type=int, default=CONFIG["max_ep"])
    args = ap.parse_args()
    t0 = time.time()
    cfg = CONFIG

    tag = args.tag or f"full_{args.crop}_f{args.fold}_s{args.seed}"
    oof_path = ART / "oof" / f"{tag}.npz"
    ckpt_path = ART / "ckpt" / f"{tag}.pt"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if oof_path.exists():
        print(f"[gruT] {tag}: oof exists, nothing to do"); return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed * 1000 + args.fold)

    wids = sorted(p.stem for p in FEAT_DIR.glob("*.npz"))
    store = WellStore(wids)
    fm = fold_map(wids)
    trainval = [w for w in wids if fm[w] != args.fold]
    test = [w for w in wids if fm[w] == args.fold]
    sw = np.array(sorted(trainval))
    tr_i, iv_i = next(GroupKFold(cfg["inner_k"]).split(sw, groups=sw))
    tr, iv = list(sw[tr_i]), list(sw[iv_i])

    model = RefinerGRU(store.n_ch, cfg["hidden"], cfg["drop"]).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    def lr_at(ep):
        if ep < cfg["warmup_ep"]:
            return cfg["lr"] * (ep + 1) / cfg["warmup_ep"]
        t = (ep - cfg["warmup_ep"]) / max(args.max_ep - cfg["warmup_ep"], 1)
        return 1e-5 + 0.5 * (cfg["lr"] - 1e-5) * (1 + np.cos(np.pi * t))

    start_ep, best_val, best_state, bad = 0, np.inf, None, 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_ep, best_val, bad = ck["epoch"] + 1, ck["best_val"], ck["bad"]
        best_state = ck["best_state"]
        torch.set_rng_state(ck["rng_torch"].cpu().to(torch.uint8))
        if device == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"].cpu().to(torch.uint8))
        print(f"[gruT] {tag}: resumed at epoch {start_ep} (best {best_val:.3f})")

    print(f"[gruT] {tag}: device={device} params={n_par} tr={len(tr)} iv={len(iv)} "
          f"test={len(test)} n_ch={store.n_ch} crop={args.crop}")

    for ep in range(start_ep, args.max_ep):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep)
        rng = np.random.default_rng(args.seed * 100000 + args.fold * 1000 + ep)
        model.train()
        tot, nb = 0.0, 0
        items = epoch_items(store, tr, rng, use_crops=(args.crop == "on"))
        for Xb, yb, mb, Ls in batches(store, items, cfg["batch"], rng, train_aug=True):
            X = torch.from_numpy(Xb).to(device); y = torch.from_numpy(yb).to(device)
            m = torch.from_numpy(mb).to(device)
            pred = model(X, torch.from_numpy(Ls))
            loss = masked_huber(pred, y, m, cfg["huber_delta"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tot += float(loss); nb += 1
        val = pooled_rmse(store, predict_wells(model, store, iv, device))
        improved = val < best_val - 1e-4
        if improved:
            best_val, bad = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"[gruT] {tag} ep{ep:03d} loss={tot / max(nb, 1):.4f} "
              f"iv_pooled={val:.4f} best={best_val:.4f} bad={bad}", flush=True)
        if bad >= cfg["patience"]:
            print(f"[gruT] {tag}: early stop at ep{ep}"); break
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep,
                        best_val=best_val, best_state=best_state, bad=bad,
                        rng_torch=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state() if device == "cuda" else None),
                   ckpt_path)
        if args.minutes > 0 and (time.time() - t0) / 60.0 > args.minutes:
            print(f"[gruT] {tag}: time budget hit at ep{ep}, exiting for resume")
            sys.exit(3)

    model.load_state_dict(best_state)
    preds = predict_wells(model, store, test, device)
    test_pooled = pooled_rmse(store, preds)
    np.savez_compressed(oof_path, **{w: r.astype(np.float32) for w, r in preds.items()})
    # keep the final weights for kernel deployment (15-model mean on test wells)
    wdir = ART / "weights"; wdir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, wdir / f"{tag}.pt")
    with open(ART / "runs.jsonl", "a") as f:
        f.write(json.dumps(dict(tag=tag, fold=args.fold, seed=args.seed,
                                crop=args.crop, best_iv=best_val,
                                test_pooled=test_pooled, params=n_par,
                                minutes=round((time.time() - t0) / 60, 1))) + "\n")
    ckpt_path.unlink(missing_ok=True)
    print(f"[gruT] {tag}: DONE best_iv={best_val:.4f} test_pooled={test_pooled:.4f}")


if __name__ == "__main__":
    main()
