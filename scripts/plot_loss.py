"""Plot loss/metric curves from a run's local CSVLogger metrics.csv.

Usage: python scripts/plot_loss.py <run_dir>
"""
import sys, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

run_dir = sys.argv[1].rstrip("/")
csvs = sorted(glob.glob(f"{run_dir}/metrics/version_*/metrics.csv"))
assert csvs, f"no metrics.csv under {run_dir}/metrics/version_*/"
csv = csvs[-1]
print("reading", csv)

df = pd.read_csv(csv)
step_col = "step" if "step" in df.columns else df.columns[0]
metric_keys = [c for c in df.columns
               if any(t in c.lower() for t in ("loss", "mse", "psnr", "ssim", "uiqi", "fvd", "lpips"))]
print("step_col:", step_col, "| metrics:", metric_keys)

if metric_keys:
    plt.figure(figsize=(11, 2.8 * len(metric_keys)))
    for i, mk in enumerate(metric_keys, 1):
        sub = df[[step_col, mk]].dropna()
        if sub.empty:
            continue
        ax = plt.subplot(len(metric_keys), 1, i)
        ax.plot(sub[step_col], sub[mk], lw=1, marker=".", ms=3)
        ax.set_title(mk)
        ax.set_xlabel(step_col)
        ax.grid(alpha=0.3)
        print(f"  {mk}: {len(sub)} pts, first={sub[mk].iloc[0]:.4f} last={sub[mk].iloc[-1]:.4f} min={sub[mk].min():.4f}")
    plt.tight_layout()
    out = f"{run_dir}/loss_curves.png"
    plt.savefig(out, dpi=90)
    print("wrote", out)
else:
    print("no metric columns found")
