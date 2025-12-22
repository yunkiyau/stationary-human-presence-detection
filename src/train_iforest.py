"""
Train Isolation Forest on FFT-bin features from the development dataset.

Per-file feature:
- Read CSV, take ALL COLUMNS from the 5th column onward (iloc[:, 4:]).
- Coerce to numeric; if multiple rows, average rows → one vector per file.
- Per-file L2-normalize (focus on spectral shape).

Pipeline:
- StandardScaler → PCA (variance target or n comps; whiten=True) [optional via flags] → IsolationForest
- Training set:
    * If dev_negdir provided and matches any files: TRAIN ON NEGATIVES ONLY (fully unsupervised).
    * Else: train on ALL dev files and rely on `--contamination`.

Outputs:
- joblib: {"scaler","pca","iforest","feature_from_col":int}
- dev_predictions.csv (file, score, if_pred(±1), pred_label(0/1), [true_label if available])
- dev_training_summary.html

Usage:
  python train_iforest.py \
    --dev_dir ./dev_raw \
    --dev_posdir ./dev_pos --dev_negdir ./dev_neg \
    --pca_var 0.95 \
    --n_estimators 200 \
    --contamination 0.10 \
    --out_model iforest_model.joblib \
    --dev_pred dev_predictions_if.csv \
    --summary_html dev_summary_if.html
"""

from pathlib import Path
import argparse, sys, html
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from datetime import datetime
import joblib
import matplotlib.pyplot as plt

# ---------------- Helpers ----------------
def list_csvs(folder: Path, recursive: bool) -> list[Path]:
   """
    List CSV files under a folder.

    Parameters
    ----------
    folder : Path
        Root folder to search.
    recursive : bool
        If True, search recursively with rglob; otherwise only direct children.

    Returns
    -------
    list[Path]
        Sorted list of CSV file paths.
    """
    if recursive:
        return sorted(p for p in folder.rglob("*.csv") if p.is_file())
    return sorted(p for p in folder.glob("*.csv") if p.is_file())

def load_fft_vector(csv_path: Path, from_col: int = 4) -> np.ndarray | None:
     """
    Load one CSV file and return a single FFT-bin feature vector.

    The script expects FFT bins to start at column index `from_col`. If the CSV has
    multiple rows (e.g., multiple frames), it averages them to produce one vector.

    Steps:
    - Read CSV (with header; fallback to header=None)
    - Slice columns from from_col onward
    - Coerce to numeric; drop rows with any NaN/inf
    - Average rows -> 1D vector
    - L2-normalize (emphasize spectral shape over absolute scale)

    Parameters
    ----------
    csv_path : Path
        Path to one CSV file.
    from_col : int
        0-indexed column index where FFT bins begin.

    Returns
    -------
    np.ndarray | None
        1D normalized feature vector, or None if the file cannot be parsed.
    """
   
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
    # per-sample L2 normalise (focus on spectral shape)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def build_name_sets(folder: Path | None) -> tuple[set[str], set[str]]::
   """
    Build fast lookup sets for matching dev files to pos/neg ground-truth folders.

    The matching is done by:
    - full filename (case-insensitive)
    - stem (filename without extension)

    Parameters
    ----------
    folder : Path | None
        Folder containing CSV files for one class.

    Returns
    -------
    tuple[set[str], set[str]]
        (filenames_set, stems_set), both lowercased.
    """
   
    files_set, stems_set = set(), set()
    if folder and folder.exists():
        for p in folder.iterdir():
            if p.is_file():
                files_set.add(p.name.lower())
                stems_set.add(p.stem.lower())
    return files_set, stems_set

def infer_label_by_dirs(path: Path, pos_sets, neg_sets) -> int | None:
   """
    Infer ground-truth label for a file based on membership in pos/neg directories.

    Returns:
    - 1 for positive class (human present)
    - 0 for negative class (no human)
    - None if unknown or ambiguous (matches both or neither)

    Parameters
    ----------
    path : Path
        File to label.
    pos_sets : tuple[set[str], set[str]]
        (pos_filenames, pos_stems) from build_name_sets().
    neg_sets : tuple[set[str], set[str]]
        (neg_filenames, neg_stems) from build_name_sets().

    Returns
    -------
    int | None
        Inferred label or None if unknown/ambiguous.
    """

    """Return 1 (pos), 0 (neg), or None."""
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

def fit_pca(X, pca_components: int | None, pca_var: float | None) -> PCA | None:
   """
    Fit PCA on training data if PCA settings are provided.

    Parameters
    ----------
    X : np.ndarray
        Training data after scaling.
    pca_components : int | None
        If set, use an explicit number of components.
    pca_var : float | None
        If set (e.g., 0.95), choose n_components to preserve this explained variance.

    Returns
    -------
    PCA | None
        Fitted PCA instance, or None if PCA disabled.
    """

    if pca_var is not None:
        return PCA(n_components=pca_var, whiten=True, svd_solver="full", random_state=42).fit(X)
    if pca_components is not None:
        return PCA(n_components=pca_components, whiten=True, svd_solver="full", random_state=42).fit(X)
    return None  # No PCA if neither provided

def cm_metrics(cm):
    """
    Compute standard binary classification metrics from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix with shape (2,2) ordered as labels [0,1].

    Returns
    -------
    dict
        accuracy, sensitivity (TPR), specificity (TNR), precision (PPV), f1, youden_J.
    """
   
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
   """
    Write a small HTML report summarising training configuration and dev performance.

    Parameters
    ----------
    path : Path
        Output HTML filepath.
    ctx : dict
        Dictionary of summary fields (strings/numbers/paths).

    Returns
    -------
    None
    """

    esc = {k: html.escape(str(v)) for k, v in ctx.items()}
    html_str = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DEV IsolationForest Summary</title>
<style>
body {{ font-family: system-ui, Segoe UI, Roboto, Arial; margin:24px; }}
table {{ border-collapse: collapse; margin: 10px 0; }}
th, td {{ border:1px solid #ddd; padding:6px 10px; text-align:right; }}
th {{ background:#f6f6f6; }} td.label {{ text-align:left; }}
.card {{ border:1px solid #e6e6e6; border-radius:8px; padding:12px; margin:8px 0; }}
.muted {{ color:#666; }}
</style></head><body>
<h1>Development Training Summary (Isolation Forest)</h1>
<div class="muted">{esc['timestamp']}</div>

<div class="card">Dev dir: <code>{esc['dev_dir']}</code><br/>
Pos/Neg dirs: <code>{esc['dev_posdir']}</code> | <code>{esc['dev_negdir']}</code><br/>
Files used: {esc['n_used']} / {esc['n_total']} (skipped {esc['n_skipped']})<br/>
Training rows: {esc['n_train']}
</div>

<div class="card">Scaler: StandardScaler • PCA comps: {esc['pca_n']} (whiten={esc['pca_whiten']})<br/>
IsolationForest: n_estimators={esc['n_estimators']}, contamination={esc['contamination']}, max_features={esc['max_features']}
</div>

<div class="card"><b>Dev metrics (if GT available)</b><br/>
<table>
<tr><th></th><th>PRED 0</th><th>PRED 1</th></tr>
<tr><td class="label">TRUE 0 (neg)</td><td>{esc['tn']}</td><td>{esc['fp']}</td></tr>
<tr><td class="label">TRUE 1 (pos)</td><td>{esc['fn']}</td><td>{esc['tp']}</td></tr>
</table>
Accuracy: {esc['acc']} • TPR: {esc['tpr']} • TNR: {esc['tnr']} • F1: {esc['f1']} • J: {esc['J']}
</div>
</body></html>"""
    path.write_text(html_str, encoding="utf-8")

# ---------------- Main ----------------
def main():
   """Train an Isolation Forest on development FFT-bin features and write summary outputs."""
    ap = argparse.ArgumentParser(
        description="Train Isolation Forest on FFT-bin vectors (with optional PCA) and export dev summaries."
    )
    ap.add_argument("--dev_dir", type=Path, required=True, help="Directory containing dev CSV files.")
    ap.add_argument("--recursive", action="store_true", help="Recursively search for CSV files under dev_dir.")
    ap.add_argument("--dev_posdir", type=Path, default=None, help="Optional directory of dev positive examples.")
    ap.add_argument("--dev_negdir", type=Path, default=None, help="Optional directory of dev negative examples.")
    ap.add_argument("--pca_components", type=int, default=None, help="If set, use this many PCA components.")
    ap.add_argument("--pca_var", type=float, default=None, help="e.g., 0.95 (omit to disable PCA)")
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--contamination", type=float, default=0.10, help="Expected fraction of anomalies.")
    ap.add_argument("--max_features", type=float, default=1.0, help="IsolationForest max_features.")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--out_model", type=Path, default=Path("iforest_model.joblib"))
    ap.add_argument("--dev_pred", type=Path, default=Path("dev_predictions_if.csv"))
    ap.add_argument("--summary_html", type=Path, default=Path("dev_summary_if.html"))
    ap.add_argument("--from_col", type=int, default=4, help="Start column for FFT bins (0-index).")
    ap.add_argument("--pca2d_png", type=Path, default=Path("dev_pca2d.png"))
    ap.add_argument("--make_pca2d_plot", action="store_true", help="Write PCA(2) scatter colored by anomaly score.")
    args = ap.parse_args()

    dev_files = list_csvs(args.dev_dir, args.recursive)
    X_all, names = [], []
    for f in dev_files:
        v = load_fft_vector(f, from_col=args.from_col)
        if v is not None:
            X_all.append(v); names.append(f)
    if not X_all:
        print("ERROR: No usable dev vectors found.", file=sys.stderr); sys.exit(1)
    X_all = np.vstack(X_all)

    # Optional GT (for summary/metrics and selecting negatives for training)
    y_true = None
    pos_sets = neg_sets = None
    if args.dev_posdir and args.dev_negdir:
        pos_sets = build_name_sets(args.dev_posdir)
        neg_sets = build_name_sets(args.dev_negdir)
        gt = [infer_label_by_dirs(p, pos_sets, neg_sets) for p in names]
        if any(g is not None for g in gt):
            y_true = np.array([np.nan if g is None else g for g in gt], dtype=float)

    # Choose training subset
    if y_true is not None and np.isfinite(y_true).any():
        train_mask = (y_true == 0)  # negatives only
        if train_mask.sum() == 0:
            print("[WARN] No matched negatives; training on all dev files.", file=sys.stderr)
            X_train = X_all
        else:
            X_train = X_all[train_mask]
    else:
        print("[INFO] No GT: training on all dev files; rely on --contamination.", file=sys.stderr)
        X_train = X_all

    # Scale + PCA
    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)
    pca = fit_pca(Xs_train, args.pca_components, args.pca_var)
    Xr_train = pca.transform(Xs_train) if pca is not None else Xs_train

    # Fit Isolation Forest
    iforest = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        max_features=args.max_features,
        random_state=args.random_state,
    ).fit(Xr_train)

    # Save model bundle
    joblib.dump({
        "scaler": scaler,
        "pca": pca,
        "iforest": iforest,
        "feature_from_col": args.from_col
    }, args.out_model)

    # Predict on all dev for summary
    Xs_all = scaler.transform(X_all)
    Xr_all = pca.transform(Xs_all) if pca is not None else Xs_all

   # sklearn IsolationForest.predict: +1=inlier, -1=outlier
    if_pred = iforest.predict(Xr_all)  # +1 inlier (NEG), -1 outlier (POS)
    # map to {0,1}

   # Map: inlier -> 0 (non-human), outlier -> 1 (human-present anomaly)
    pred_label = np.where(if_pred == 1, 0, 1)

    # anomaly score: higher = more anomalous
    scores = -iforest.decision_function(Xr_all)

    if args.make_pca2d_plot:
        Z = PCA(n_components=2, random_state=42).fit_transform(Xs_all)
        fig, ax = plt.subplots(figsize=(7,5), dpi=200)
        sc = ax.scatter(Z[:,0], Z[:,1], c=scores, s=36, cmap="viridis")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Isolation Forest Anomaly Score", fontsize=13, fontweight='bold')
        ax.set_xlabel("PCA 1", fontweight="bold"); ax.set_ylabel("PCA 2", fontweight="bold")
        #ax.set_title("Development set — PCA projection colored by anomaly score", fontweight="bold")
        fig.tight_layout(); fig.savefig(args.pca2d_png, bbox_inches="tight")
        print(f"PCA 2D plot: {args.pca2d_png}")


    rows = {"file": [p.name for p in names], "score_anom": scores,
            "if_pred_pm1": if_pred, "pred_label": pred_label}
    if y_true is not None:
        rows["true_label"] = [int(t) if np.isfinite(t) else "" for t in y_true]
    pd.DataFrame(rows).to_csv(args.dev_pred, index=False)

    # Metrics (if GT available)
    tn=fp=fn=tp=""; acc=tpr=tnr=f1=J=""
    if y_true is not None and np.isfinite(y_true).any():
        m = np.isfinite(y_true)
        cm = confusion_matrix(y_true[m].astype(int), pred_label[m], labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        met = cm_metrics(cm)
        acc=f"{met['accuracy']:.3f}"; tpr=f"{met['sensitivity']:.3f}"
        tnr=f"{met['specificity']:.3f}"; f1=f"{met['f1']:.3f}"; J=f"{met['youden_J']:.3f}"

    # HTML summary
    ctx = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dev_dir": args.dev_dir, "dev_posdir": args.dev_posdir, "dev_negdir": args.dev_negdir,
        "n_total": len(dev_files), "n_used": len(names), "n_skipped": len(dev_files)-len(names),
        "n_train": X_train.shape[0],
        "pca_n": (pca.n_components_ if pca is not None else "None"),
        "pca_whiten": (pca.whiten if pca is not None else False),
        "n_estimators": args.n_estimators,
        "contamination": args.contamination,
        "max_features": args.max_features,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "acc": acc, "tpr": tpr, "tnr": tnr, "f1": f1, "J": J
    }
    write_html(args.summary_html, ctx)

    print(f"Trained IsolationForest. Model saved to {args.out_model}")
    print(f"Dev predictions CSV: {args.dev_pred}")
    print(f"Summary HTML: {args.summary_html}")

if __name__ == "__main__":
    main()


