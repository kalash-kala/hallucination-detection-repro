"""
Experiment 3 — Fused sink_score x ||value_norm|| feature (Steps A-D).

feature_3[l, h, rank] = s_i * ||V_i||  where, per (layer l, query-head h):
  - positions are ranked by sink score s_i = D = attn_diag + lap_diag (from cache),
  - top-K=10 ranks are kept (the Exp-3 feature width, matching TOP_K_LAP),
  - ||V_i|| is the per-KV-head value-vector norm (extract_value_norms.py); query head
    h reads KV head h // (n_q_heads // n_kv_heads).
Shape [L, H, K] -> flattened, identical to the LapEigvals top-k feature, so it is a
true drop-in replacement for the lap half of the lap+hidden combo.

Validation sequence (spec Section 4):
  A  sanity: computable, finite, non-constant, ||V|| below hidden-norm scale.
  B  standalone per-CELL Mann-Whitney heatmaps for the 4 band contrasts (NO global
     flatten); report % BH-significant cells per contrast.
  C  same per-cell reduction for lap_only, same cells/contrasts; gate to D.
  D  full classifier: swap lap -> feature_3 in the lap+hidden combo, identical
     hidden half, StandardScaler->LR; report all 4 band columns (IHvC/ILvC/CHvI/CLvI)
     vs lap+hidden on identical ids, against the Section-1 success criteria.

Usage:
  PY=/root/miniconda3/envs/semantic_uncertainty/bin/python
  $PY scripts/sinkhole/exp3_feature.py --all
  $PY scripts/sinkhole/exp3_feature.py --model llama --dataset sciq --step B
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import false_discovery_control, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "feature_analysis"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lapeigvals_baseline"))
import train_2axis as TA                                   # noqa: E402
from lapeigvals_features import get_laplacian_eigvals_per_head_topk  # noqa: E402

DIAG_DIR = REPO_ROOT / "results" / "lapeigvals_baseline" / "diags"
VN_DIR = REPO_ROOT / "results" / "sinkhole" / "value_norms"
OUT_DIR = REPO_ROOT / "results" / "sinkhole" / "exp3"

PAIRS = [(m, d) for m in ("llama", "mistral", "qwen", "gemma")
         for d in ("sciq", "triviaqa", "math")]
K = TA.TOP_K_LAP            # 10
TOP5 = 5                    # ranks used for the per-cell Step-B/C summary
SEED = 42
ALPHA = 0.05
LR_KWARGS = dict(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)
CAT_SHORT = {"incorrect_high": "IH", "incorrect_low": "IL",
             "correct_high": "CH", "correct_low": "CL"}
INCORRECT = {"IH", "IL"}
CORRECT = {"CH", "CL"}
# band contrasts: (column, positive cat). Negatives = opposite correctness pool.
CONTRASTS = [("IHvC", "IH"), ("ILvC", "IL"), ("CHvI", "CH"), ("CLvI", "CL")]


# ── feature construction ─────────────────────────────────────────────────────

def build_pair(model: str, dataset: str, split: str, k: int):
    """Return per-id arrays: feature_3 [L*H*k], cell-summary [L,H], lap [L*H*k],
    lap cell-summary [L,H]; plus labels, cats, (L,H)."""
    dg = torch.load(DIAG_DIR / f"{model}_{dataset}_{split}.pt", weights_only=False)
    vnf = torch.load(VN_DIR / f"{model}_{dataset}_{split}.pt", weights_only=False)
    nq, nkv = vnf["n_q_heads"], vnf["n_kv_heads"]
    group = nq // nkv
    kv_of_head = np.arange(nq) // group        # [H]

    vn_by = {i: vnf["value_norms"][j] for j, i in enumerate(vnf["ids"])}
    lab_by = {i: int(vnf["labels"][j]) for j, i in enumerate(vnf["ids"])}

    lap_diags = [d.float() for d in dg["lap_diags"]]
    attn_diags = [d.float() for d in dg["attn_diags"]]
    ids = [str(i) for i in dg["ids"]]
    cats = dg["categories"]
    # lap top-k feature, byte-identical to the baseline
    lap_X = get_laplacian_eigvals_per_head_topk(lap_diags, top_k=k).numpy()  # [N, L*H*k]
    lap_full_sorted = [torch.sort(d, dim=-1, descending=True).values for d in lap_diags]

    f3, f3cell, lapv, lapcell, labels, cat_out, out_ids = {}, {}, {}, {}, {}, {}, []
    sink = {}
    L = H = None
    for n, sid in enumerate(ids):
        if sid not in vn_by:
            continue
        D = (attn_diags[n] + lap_diags[n]).numpy()         # [L,H,S] sink score
        vn = vn_by[sid].float().numpy()                    # [L,nkv,S]
        if D.shape[-1] != vn.shape[-1]:
            continue
        Lh, Hh, S = D.shape
        if L is None:
            L, H = Lh, Hh
        kk = min(k, S)
        idx = np.argsort(-D, axis=-1)[:, :, :kk]            # [L,H,kk] top by sink
        d_top = np.take_along_axis(D, idx, axis=-1)         # [L,H,kk]
        vn_h = vn[:, kv_of_head, :]                         # [L,H,S] broadcast kv->q
        vn_top = np.take_along_axis(vn_h, idx, axis=-1)     # [L,H,kk]
        feat = d_top * vn_top                               # [L,H,kk]
        sink_feat = d_top                                   # [L,H,kk] sink score only
        if kk < k:                                          # pad short examples
            feat = np.pad(feat, ((0, 0), (0, 0), (0, k - kk)), mode="edge")
            sink_feat = np.pad(sink_feat, ((0, 0), (0, 0), (0, k - kk)), mode="edge")
        f3[sid] = feat.reshape(-1).astype(np.float32)
        sink[sid] = sink_feat.reshape(-1).astype(np.float32)
        f3cell[sid] = feat[:, :, :TOP5].mean(axis=-1).astype(np.float32)   # [L,H]
        lapv[sid] = lap_X[n].astype(np.float32)
        ls = lap_full_sorted[n].numpy()                     # [L,H,S] sorted desc
        lapcell[sid] = ls[:, :, :TOP5].mean(axis=-1).astype(np.float32)
        labels[sid] = lab_by.get(sid, int(dg["labels"][n]))
        cat_out[sid] = CAT_SHORT.get(cats[n], "?")
        out_ids.append(sid)
    return dict(ids=out_ids, f3=f3, sink=sink, f3cell=f3cell, lap=lapv, lapcell=lapcell,
                labels=labels, cats=cat_out, L=L, H=H)


# ── Step A ───────────────────────────────────────────────────────────────────

def step_A(pair, tr, te, report):
    allf = np.stack([tr["f3"][i] for i in tr["ids"]] + [te["f3"][i] for i in te["ids"]])
    finite = np.isfinite(allf).all()
    const = float(allf.std()) == 0.0
    line = (f"[A] {pair}: dim={allf.shape[1]} finite={finite} non_constant={not const} "
            f"range=[{allf.min():.3g},{allf.max():.3g}] mean={allf.mean():.3g}")
    print("  " + line); report.append(line)
    return finite and not const


# ── Step B / C per-cell Mann-Whitney ─────────────────────────────────────────

def cell_pvals(cell_by, ids, cats, pos, L, H):
    """MWU per (l,h) cell between positive cat and opposite-correctness pool."""
    cats = np.array([cats[i] for i in ids])
    neg = CORRECT if pos in INCORRECT else INCORRECT
    mpos = cats == pos
    mneg = np.isin(cats, list(neg))
    if mpos.sum() < 5 or mneg.sum() < 5:
        return None
    M = np.stack([cell_by[i] for i in ids])        # [N, L, H]
    A = M[mpos].reshape(mpos.sum(), -1)            # [npos, L*H]
    Bn = M[mneg].reshape(mneg.sum(), -1)
    p = np.ones(A.shape[1])
    for c in range(A.shape[1]):
        a, b = A[:, c], Bn[:, c]
        if np.ptp(np.concatenate([a, b])) == 0:
            continue
        try:
            p[c] = mannwhitneyu(a, b, alternative="two-sided").pvalue
        except ValueError:
            continue
    return p


def step_BC(pair, te, report, heat_dir):
    L, H = te["L"], te["H"]
    rows = []
    for col, pos in CONTRASTS:
        for which, cell_key in (("feature3", "f3cell"), ("lap_only", "lapcell")):
            p = cell_pvals(te[cell_key], te["ids"], te["cats"], pos, L, H)
            if p is None:
                rows.append((pair, col, which, float("nan"), 0))
                continue
            q = false_discovery_control(p, method="bh")
            nsig = int((q < ALPHA).sum())
            pct = 100.0 * nsig / len(q)
            rows.append((pair, col, which, pct, nsig))
            np.save(heat_dir / f"{pair}_{col}_{which}_qheat.npy",
                    q.reshape(L, H))
    # compact print
    by = {(c, w): (pct, n) for (pr, c, w, pct, n) in rows}
    for col, _ in CONTRASTS:
        f = by.get((col, "feature3"), (float("nan"), 0))
        l = by.get((col, "lap_only"), (float("nan"), 0))
        line = (f"[B/C] {pair} {col}: feature3 %sig={f[0]:.1f} ({f[1]}) | "
                f"lap_only %sig={l[0]:.1f} ({l[1]})")
        print("  " + line); report.append(line)
    return rows


# ── Step D band-AUROC combo ──────────────────────────────────────────────────

def band_auroc(scores, cats, target):
    neg = CORRECT if target in INCORRECT else INCORRECT
    mask = (cats == target) | np.isin(cats, list(neg))
    y = (cats[mask] == target).astype(int)
    s = scores[mask]
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    a = float(roc_auc_score(y, s))
    return (1.0 - a) if target in CORRECT else a


def fit_combo(tr, te, feat_key, hid_tr, hid_te):
    """Fit StandardScaler->LR on [feat_key + hidden], return test (scores, cats)."""
    feat_tr, feat_te = tr[feat_key], te[feat_key]
    tr_ids = [i for i in tr["ids"] if i in hid_tr]
    te_ids = [i for i in te["ids"] if i in hid_te]
    Xtr = np.stack([np.concatenate([feat_tr[i], hid_tr[i]]) for i in tr_ids])
    Xte = np.stack([np.concatenate([feat_te[i], hid_te[i]]) for i in te_ids])
    ytr = np.array([tr["labels"][i] for i in tr_ids])
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(**LR_KWARGS))])
    pipe.fit(Xtr, ytr)
    sc = pipe.predict_proba(Xte)[:, 1]
    cats = np.array([te["cats"][i] for i in te_ids])
    return sc, cats


def step_D(model, dataset, tr, te, report, rows_out):
    pair = f"{model}_{dataset}"
    hid_tr = TA.load_hidden(model, dataset, "train")
    hid_te = TA.load_hidden(model, dataset, "test")
    res = {}
    for name, key in (("feature3+hidden", "f3"), ("lap+hidden", "lap")):
        sc, cats = fit_combo(tr, te, key, hid_tr, hid_te)
        cols = {col: band_auroc(sc, cats, pos) for col, pos in CONTRASTS}
        res[name] = cols
        rows_out.append([pair, name] + [cols[c] for c, _ in CONTRASTS])
    f, l = res["feature3+hidden"], res["lap+hidden"]
    line = (f"[D] {pair}\n"
            f"      {'':16s}  IHvC    ILvC    CHvI    CLvI\n"
            f"      feature3+hidden  {f['IHvC']:.4f}  {f['ILvC']:.4f}  {f['CHvI']:.4f}  {f['CLvI']:.4f}\n"
            f"      lap+hidden       {l['IHvC']:.4f}  {l['ILvC']:.4f}  {l['CHvI']:.4f}  {l['CLvI']:.4f}\n"
            f"      delta            {f['IHvC']-l['IHvC']:+.4f}  {f['ILvC']-l['ILvC']:+.4f}  "
            f"{f['CHvI']-l['CHvI']:+.4f}  {f['CLvI']-l['CLvI']:+.4f}")
    print(line); report.append(line)
    return res


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama", "mistral", "qwen", "gemma"])
    ap.add_argument("--dataset", choices=["sciq", "triviaqa", "math"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--step", default="ABCD", help="subset of steps to run, e.g. 'AB' or 'D'")
    args = ap.parse_args()
    pairs = PAIRS if args.all else [(args.model, args.dataset)]
    if not args.all and not (args.model and args.dataset):
        raise SystemExit("specify --all or both --model and --dataset")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    heat_dir = OUT_DIR / "heatmaps"; heat_dir.mkdir(exist_ok=True)
    report = ["# Experiment 3 — Fused sink_score x ||value_norm|| feature", "",
              f"K={K} feature width; per-cell summary = mean over top-{TOP5} ranks; "
              f"BH FDR={ALPHA}.", ""]
    bc_rows, d_rows = [], []

    for model, dataset in pairs:
        pair = f"{model}_{dataset}"
        if not (VN_DIR / f"{model}_{dataset}_test.pt").exists():
            print(f"[skip] {pair}: value norms not extracted yet"); continue
        print(f"\n=== {pair} ===")
        try:
            tr = build_pair(model, dataset, "train", K)
            te = build_pair(model, dataset, "test", K)
            report.append(f"## {pair}")
            if "A" in args.step:
                step_A(pair, tr, te, report)
            if "B" in args.step or "C" in args.step:
                bc_rows += step_BC(pair, te, report, heat_dir)
            if "D" in args.step:
                step_D(model, dataset, tr, te, report, d_rows)
            report.append("")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {pair} failed: {e}")
            traceback.print_exc()
            report.append(f"## {pair}\n[ERROR] {e}\n")

    if bc_rows:
        with (OUT_DIR / "stepBC_cell_significance.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["pair", "contrast", "feature", "pct_sig", "n_sig"])
            w.writerows(bc_rows)
    if d_rows:
        with (OUT_DIR / "stepD_band_auroc.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["pair", "method", "IHvC", "ILvC", "CHvI", "CLvI"])
            w.writerows(d_rows)
        # pooled means + success criteria
        report.append("## Pooled mean across pairs (Step D)")
        import collections
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for pr, meth, ih, il, ch, cl in d_rows:
            for c, v in zip(("IHvC", "ILvC", "CHvI", "CLvI"), (ih, il, ch, cl)):
                if not np.isnan(v):
                    agg[meth][c].append(v)
        means = {m: {c: float(np.mean(agg[m][c])) for c in ("IHvC", "ILvC", "CHvI", "CLvI")}
                 for m in agg}
        for m, cols in means.items():
            report.append(f"- {m}: IHvC={cols['IHvC']:.4f} ILvC={cols['ILvC']:.4f} "
                          f"CHvI={cols['CHvI']:.4f} CLvI={cols['CLvI']:.4f}")
        if "feature3+hidden" in means and "lap+hidden" in means:
            f, l = means["feature3+hidden"], means["lap+hidden"]
            d = {c: f[c] - l[c] for c in f}
            improved = (d["IHvC"] > 0) or (d["CLvI"] > 0)
            no_regress = (d["ILvC"] >= -0.01) and (d["CHvI"] >= -0.01)
            verdict = ("PASS" if (improved and no_regress) else "FAIL")
            report.append("")
            report.append(f"**Success criteria (pooled): {verdict}** — "
                          f"deltas IHvC={d['IHvC']:+.4f} CLvI={d['CLvI']:+.4f} "
                          f"(improve target) | ILvC={d['ILvC']:+.4f} CHvI={d['CHvI']:+.4f} "
                          f"(>=-0.01 guard).")

    (OUT_DIR / "EXP3_REPORT.md").write_text("\n".join(report))
    print(f"\nWrote {OUT_DIR/'EXP3_REPORT.md'}")


if __name__ == "__main__":
    main()