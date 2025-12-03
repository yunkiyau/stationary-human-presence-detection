#!/usr/bin/env python3
"""
Analyze development-set features and pick Youden-optimal thresholds.

What it does
------------
- Loads dev_neg_features.csv (label=0) and dev_pos_features.csv (label=1)
- For each numeric feature:
    * Cleans NaNs/inf
    * Plots histograms (neg vs pos) with the chosen threshold marked
    * Searches midpoints between sorted unique values as candidate cutoffs
      for BOTH directions (>= and <=)
    * Picks the rule that maximizes Youden's J (TPR - FPR)
    * Prints confusion matrix + metrics (TPR, FPR, Accuracy, J)
- Writes a summary CSV with the recommended threshold per feature.

Usage
-----
python development_thresholds.py 
  --neg /path/to/dev_neg_features.csv 
  --pos /path/to/dev_pos_features.csv 
  --out_csv /path/to/dev_thresholds_summary.csv 
  --plots_dir /path/to/dev_feature_plots

Notes
-----
- By default it looks for common feature columns:
    spectral_flatness, crest_factor, spectral_centroid_Hz, total_power
  If some are missing, it automatically falls back to ALL numeric columns
  (excluding identifiers/metadata).
- The script is robust to minor column differences and NaNs.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


DEFAULT_FEATURES = [
    "spectral_flatness",
    "crest_factor",
    "spectral_centroid_Hz",
    "total_power",
]

# --- Pretty display names for plots / messages ---
NAME_MAP = {
    "spectral_flatness": "Spectral flatness",
    "crest_factor": "Crest factor",
    "spectral_centroid_Hz": "Spectral centroid (Hz)",
    "total_power": "Phase variance",
}
META_COLS = {
    "file", "label", "fs_Hz", "n_samples", "f_min_Hz", "f_max_Hz",
    "error", "phase_col", "time_col"
}

def disp(feat: str) -> str:
    """Return a pretty display name for a feature key."""
    # EXACTLY what you asked for: look up the key, else nice fallback
    return NAME_MAP.get(feat, feat.replace("_", " ").title())

def slug(text: str) -> str:
    """File/FS-safe slug from a display name."""
    return (text.lower()
            .replace("(", "").replace(")", "")
            .replace("[", "").replace("]", "")
            .replace("%", "pct")
            .replace(" ", "_").replace("/", "_"))

def load_and_label(neg_csv: Path, pos_csv: Path) -> pd.DataFrame:
    df_neg = pd.read_csv(neg_csv); df_neg["label"] = 0
    df_pos = pd.read_csv(pos_csv); df_pos["label"] = 1
    return pd.concat([df_neg, df_pos], ignore_index=True)

def pick_features(df: pd.DataFrame) -> list[str]:
    feats = [c for c in DEFAULT_FEATURES if c in df.columns]
    if feats:
        return feats
    # fallback: ALL numeric, excluding obvious metadata
    return [c for c in df.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]

def clean_df(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in features:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=features + ["label"])

def youden_optimal_threshold(y_true: np.ndarray, x: np.ndarray) -> dict:
    order = np.argsort(x); xs = x[order]; ys = y_true[order]
    uniq = np.unique(xs)

    if uniq.size == 1:
        # Degenerate: every value equal → any rule yields same predictions
        tp = int((ys==1).sum()); tn = int((ys==0).sum()); fp = fn = 0
        tpr = tp/(tp+fn) if (tp+fn)>0 else 0.0
        fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
        acc = (tp+tn)/len(ys) if len(ys) else 0.0
        return dict(threshold=float(uniq[0]), direction=">=", J=tpr-fpr,
                    TPR=tpr, FPR=fpr, confusion=(tp,fp,tn,fn), accuracy=acc)

    mids = (uniq[:-1] + uniq[1:]) / 2.0
    best = None
    for thr, direction in product(mids, (">=","<=")):
        y_pred = (x >= thr).astype(int) if direction==">=" else (x <= thr).astype(int)
        TP = int(((y_pred==1)&(y_true==1)).sum())
        TN = int(((y_pred==0)&(y_true==0)).sum())
        FP = int(((y_pred==1)&(y_true==0)).sum())
        FN = int(((y_pred==0)&(y_true==1)).sum())
        TPR = TP/(TP+FN) if (TP+FN)>0 else 0.0
        FPR = FP/(FP+TN) if (FP+TN)>0 else 0.0
        J = TPR - FPR
        acc = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0.0
        cand = dict(threshold=float(thr), direction=direction, J=float(J),
                    TPR=float(TPR), FPR=float(FPR),
                    confusion=(TP,FP,TN,FN), accuracy=float(acc))
        if (best is None) or (cand["J"] > best["J"]) or (cand["J"] == best["J"] and cand["accuracy"] > best["accuracy"]):
            best = cand
    return best

def plot_histogram(df: pd.DataFrame, feat_key: str, feat_display: str, thr: float, plots_dir: Path):
    dat0 = df[df.label==0][feat_key].to_numpy()
    dat1 = df[df.label==1][feat_key].to_numpy()
    if dat0.size == 0 or dat1.size == 0:
        return

    bins = np.histogram_bin_edges(np.concatenate([dat0, dat1]), bins="auto")

    plt.figure()
    plt.hist(dat0, bins=bins, alpha=0.5, label="Negative (no human)")
    plt.hist(dat1, bins=bins, alpha=0.5, label="Positive (human)")
    plt.axvline(thr, linewidth=2, label="Threshold (Youden's J)")

    # ——— Use PRETTY names here ———
    plt.xlabel(feat_display)
    plt.ylabel("Samples (no.)")
    plt.title(f"{feat_display} distribution (development dataset)")
    plt.legend()
    plt.tight_layout()

    plots_dir.mkdir(exist_ok=True, parents=True)
    fig_path = plots_dir / f"{slug(feat_display)}_hist.png"  # from display name
    plt.savefig(fig_path, dpi=144, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg", type=Path, default=Path("dev_neg_features.csv"))
    ap.add_argument("--pos", type=Path, default=Path("dev_pos_features.csv"))
    ap.add_argument("--out_csv", type=Path, default=Path("dev_thresholds_summary.csv"))
    ap.add_argument("--plots_dir", type=Path, default=Path("dev_feature_plots"))
    args = ap.parse_args()

    df = load_and_label(args.neg, args.pos)
    features = pick_features(df)
    if not features:
        raise SystemExit("No usable numeric feature columns found.")
    df = clean_df(df, features)

    rows = []
    for feat_key in features:
        feat_display = disp(feat_key)  # pretty name once, reuse everywhere

        y = df["label"].to_numpy(dtype=int)
        x = df[feat_key].to_numpy(dtype=float)
        best = youden_optimal_threshold(y, x)

        mu0 = float(df[df.label==0][feat_key].mean())
        sd0 = float(df[df.label==0][feat_key].std(ddof=1))
        mu1 = float(df[df.label==1][feat_key].mean())
        sd1 = float(df[df.label==1][feat_key].std(ddof=1))

        rows.append({
            "feature_key": feat_key,
            "feature_display": feat_display,
            "threshold": best["threshold"],
            "direction": best["direction"],
            "rule": f"{feat_display} {best['direction']} {best['threshold']:.6g} ⇒ predict HUMAN",
            "Youden_J": best["J"],
            "TPR_sensitivity": best["TPR"],
            "FPR": best["FPR"],
            "accuracy_dev": best["accuracy"],
            "TP": best["confusion"][0],
            "FP": best["confusion"][1],
            "TN": best["confusion"][2],
            "FN": best["confusion"][3],
            "neg_mean": mu0, "neg_sd": sd0,
            "pos_mean": mu1, "pos_sd": sd1,
            "n_neg": int((df.label==0).sum()),
            "n_pos": int((df.label==1).sum()),
        })

        # Plot with PRETTY names
        plot_histogram(df, feat_key=feat_key, feat_display=feat_display,
                       thr=best["threshold"], plots_dir=args.plots_dir)

    out_df = pd.DataFrame(rows).sort_values("Youden_J", ascending=False).reset_index(drop=True)
    out_df.to_csv(args.out_csv, index=False)

    print("\nRecommended single-feature thresholds (dev set):\n")
    for _, r in out_df.iterrows():
        print(
            f"- {r['feature_display']}: rule '{r['rule']}' | "
            f"TP={int(r['TP'])}, FP={int(r['FP'])}, TN={int(r['TN'])}, FN={int(r['FN'])} | "
            f"TPR={r['TPR_sensitivity']:.3f}, FPR={r['FPR']:.3f}, "
            f"Acc={r['accuracy_dev']:.3f}, J={r['Youden_J']:.3f}"
        )
    print(f"\nSaved threshold summary to: {args.out_csv}")
    print(f"Saved histograms to: {args.plots_dir.resolve()}")

if __name__ == "__main__":
    main()
