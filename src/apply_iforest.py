"""
Apply a saved Isolation Forest model to an evaluation dataset.

- Loads {"scaler","pca","iforest","feature_from_col"} from .joblib
- Extracts FFT-bin vectors from eval CSVs (from_col..end), per-file L2-normalize
- Standardize → PCA (if present) → IsolationForest.predict
- Maps +1 (inlier) → 0 (non-human), -1 (outlier) → 1 (human)
- Writes eval_predictions.csv and eval_summary_if.html

useage:
python apply_iforest.py \
  --model iforest_model.joblib \
  --eval_dir ./eval_raw \
  --eval_posdir ./eval_pos \
  --eval_negdir ./eval_neg \
  --eval_pred eval_predictions_if.csv \
  --summary_html eval_summary_if.html \
  --hist_png anomaly_hist.png \
  --hist_bins 20

"""

from pathlib import Path
import argparse, sys, html
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from datetime import datetime
import joblib

def list_csvs(folder: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(p for p in folder.rglob("*.csv") if p.is_file())
    return sorted(p for p in folder.glob("*.csv") if p.is_file())

def load_fft_vector(csv_path: Path, from_col: int = 4) -> np.ndarray | None:
    try:
        df = pd.read_csv(csv_path, header=0)
    except Exception:
        try:
            df = pd.read_csv(csv_path, header=None)
        except Exception as e2:
            print(f"[WARN] read_csv failed for {csv_path}: {e2}", file=sys.stderr)
            return None
    if df.shape[1] <= from_col:
        print(f"[WARN] {csv_path.name}: has <= {from_col} columns; no FFT bins.", file=sys.stderr)
        return None
    X = df.iloc[:, from_col:].apply(pd.to_numeric, errors="coerce").to_numpy()
    X = X[np.isfinite(X).all(axis=1)]
    if X.size == 0:
        print(f"[WARN] {csv_path.name}: FFT-bin slice all NaN/inf.", file=sys.stderr)
        return None
    v = X.mean(axis=0)
    if not np.isfinite(v).all():
        print(f"[WARN] {csv_path.name}: averaged vector has NaN/inf.", file=sys.stderr)
        return None
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def build_name_sets(folder: Path):
    files_set, stems_set = set(), set()
    if folder and folder.exists():
        for p in folder.iterdir():
            if p.is_file():
                files_set.add(p.name.lower())
                stems_set.add(p.stem.lower())
    return files_set, stems_set

def infer_label_by_dirs(path: Path, pos_sets, neg_sets) -> int | None:
    base = path.name.lower(); stem = path.stem.lower()
    pos_files, pos_stems = pos_sets; neg_files, neg_stems = neg_sets
    in_pos_file, in_neg_file = base in pos_files, base in neg_files
    if in_pos_file ^ in_neg_file:
        return 1 if in_pos_file else 0
    if in_pos_file and in_neg_file:
        return None
    in_pos_stem, in_neg_stem = stem in pos_stems, stem in neg_stems
    if in_pos_stem ^ in_neg_stem:
        return 1 if in_pos_stem else 0
    if in_pos_stem and in_neg_stem:
        return None
    return None

def cm_metrics(cm):
    tn, fp, fn, tp = cm.ravel()
    total = tn+fp+fn+tp
    acc = (tp+tn)/total if total else np.nan
    tpr = tp/(tp+fn) if (tp+fn) else np.nan
    tnr = tn/(tn+fp) if (tn+fp) else np.nan
    ppv = tp/(tp+fp) if (tp+fp) else np.nan
    f1  = (2*ppv*tpr)/(ppv+tpr) if (ppv and tpr and (ppv+tpr)) else np.nan
    fpr = fp/(fp+tn) if (fp+tn) else np.nan
    J   = tpr - fpr if (not np.isnan(tpr) and not np.isnan(fpr)) else np.nan
    return dict(accuracy=acc, sensitivity=tpr, specificity=tnr, precision=ppv, f1=f1, youden_J=J)

def write_html(path: Path, ctx: dict):
    esc = {k: html.escape(str(v)) for k, v in ctx.items()}
    html_str = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EVAL IsolationForest Summary</title>
<style>
body {{ font-family: system-ui, Segoe UI, Roboto, Arial; margin:24px; }}
table {{ border-collapse: collapse; margin: 10px 0; }}
th, td {{ border:1px solid #ddd; padding:6px 10px; text-align:right; }}
th {{ background:#f6f6f6; }} td.label {{ text-align:left; }}
.card {{ border:1px solid #e6e6e6; border-radius:8px; padding:12px; margin:8px 0; }}
.muted {{ color:#666; }}
</style></head><body>
<h1>Evaluation Summary (Isolation Forest)</h1>
<div class="muted">{esc['timestamp']}</div>

<div class="card">Eval dir: <code>{esc['eval_dir']}</code><br/>
Pos/Neg dirs: <code>{esc['eval_posdir']}</code> | <code>{esc['eval_negdir']}</code><br/>
Files used: {esc['n_used']} / {esc['n_total']} (skipped {esc['n_skipped']})
</div>

<div class="card"><b>Eval metrics (if GT available)</b><br/>
<table>
<tr><th></th><th>PRED 0</th><th>PRED 1</th></tr>
<tr><td class="label">TRUE 0 (neg)</td><td>{esc['tn']}</td><td>{esc['fp']}</td></tr>
<tr><td class="label">TRUE 1 (pos)</td><td>{esc['fn']}</td><td>{esc['tp']}</td></tr>
</table>
Accuracy: {esc['acc']} • TPR: {esc['tpr']} • TNR: {esc['tnr']} • F1: {esc['f1']} • J: {esc['J']}
</div>
</body></html>"""
    path.write_text(html_str, encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--eval_dir", type=Path, required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--eval_posdir", type=Path, default=None)
    ap.add_argument("--eval_negdir", type=Path, default=None)
    ap.add_argument("--eval_pred", type=Path, default=Path("eval_predictions_if.csv"))
    ap.add_argument("--summary_html", type=Path, default=Path("eval_summary_if.html"))
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    scaler = bundle["scaler"]; pca = bundle["pca"]; iforest = bundle["iforest"]
    from_col = int(bundle.get("feature_from_col", 4))

    eval_files = list_csvs(args.eval_dir, args.recursive)
    X_all, names = [], []
    for f in eval_files:
        v = load_fft_vector(f, from_col=from_col)
        if v is not None:
            X_all.append(v); names.append(f)
    if not X_all:
        print("ERROR: No usable eval vectors found.", file=sys.stderr); sys.exit(1)
    X_all = np.vstack(X_all)

    Xs = scaler.transform(X_all)
    Xr = pca.transform(Xs) if pca is not None else Xs

    if_pred = iforest.predict(Xr)         # +1 inlier (NEG), -1 outlier (POS)
    pred_label = np.where(if_pred == 1, 0, 1)
    scores = -iforest.decision_function(Xr)

    rows = {"file": [p.name for p in names], "score_anom": scores,
            "if_pred_pm1": if_pred, "pred_label": pred_label}
    # Optional GT for metrics
    y_true = None
    if args.eval_posdir and args.eval_negdir:
        pos_sets = build_name_sets(args.eval_posdir)
        neg_sets = build_name_sets(args.eval_negdir)
        gt = [infer_label_by_dirs(p, pos_sets, neg_sets) for p in names]
        if any(g is not None for g in gt):
            y_true = np.array([np.nan if g is None else g for g in gt], dtype=float)
            rows["true_label"] = [int(t) if np.isfinite(t) else "" for t in y_true]

    pd.DataFrame(rows).to_csv(args.eval_pred, index=False)

    # Metrics (if GT)
    tn=fp=fn=tp=""; acc=tpr=tnr=f1=J=""
    if y_true is not None and np.isfinite(y_true).any():
        m = np.isfinite(y_true)
        cm = confusion_matrix(y_true[m].astype(int), pred_label[m], labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        met = cm_metrics(cm)
        acc=f"{met['accuracy']:.3f}"; tpr=f"{met['sensitivity']:.3f}"
        tnr=f"{met['specificity']:.3f}"; f1=f"{met['f1']:.3f}"; J=f"{met['youden_J']:.3f}"

    ctx = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "eval_dir": args.eval_dir, "eval_posdir": args.eval_posdir, "eval_negdir": args.eval_negdir,
        "n_total": len(eval_files), "n_used": len(names), "n_skipped": len(eval_files)-len(names),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "acc": acc, "tpr": tpr, "tnr": tnr, "f1": f1, "J": J
    }
    write_html(args.summary_html, ctx)

    print(f"Applied IsolationForest. Predictions: {args.eval_pred}")
    print(f"Summary HTML: {args.summary_html}")

if __name__ == "__main__":
    main()

