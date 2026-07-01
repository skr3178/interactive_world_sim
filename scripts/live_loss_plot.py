"""Live loss-curve plotter (CSV-based, no wandb). Every INTERVAL seconds finds the most
recently-written metrics.csv under outputs/ and regenerates outputs/latest-run/loss_curves.png.
Runs until killed.
"""
import os, sys, glob, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = "/media/skr/storage/YC/interactive_world_sim"
OUT_DIR = f"{REPO}/outputs/latest-run"
OUT_PNG = f"{OUT_DIR}/loss_curves.png"
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 300
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ("training/rec_loss", "validation/mse", "validation/psnr",
           "validation/ssim", "validation/uiqi")
# near-perfect reconstruction targets (gate before Stage 2)
TARGETS = {"training/rec_loss": 0.001, "validation/mse": 0.001,
           "validation/psnr": 30.0, "validation/ssim": 0.90, "validation/uiqi": 0.90}


def newest_csv():
    files = glob.glob(f"{REPO}/outputs/**/metrics/version_*/metrics.csv", recursive=True)
    return max(files, key=os.path.getmtime) if files else None


def plot(csv):
    df = pd.read_csv(csv)
    step_col = "step" if "step" in df.columns else df.columns[0]
    present = [m for m in METRICS if m in df.columns and not df[m].dropna().empty]
    if not present:
        return None
    last_step = int(df[step_col].max())
    fig = plt.figure(figsize=(11, 2.6 * len(present)))
    for i, mk in enumerate(present, 1):
        sub = df[[step_col, mk]].dropna()
        ax = plt.subplot(len(present), 1, i)
        ax.plot(sub[step_col], sub[mk], lw=1, marker=".", ms=3)
        ax.set_title(f"{mk}  (last={sub[mk].iloc[-1]:.4f})")
        ax.set_xlabel(step_col)
        ax.grid(alpha=0.3)
        if mk in TARGETS:
            tgt = TARGETS[mk]
            ax.axhline(tgt, color="red", ls=":", lw=1.3, label=f"target {tgt}")
            lo, hi = min(sub[mk].min(), tgt), max(sub[mk].max(), tgt)
            pad = (hi - lo) * 0.08 or 0.001
            ax.set_ylim(lo - pad, hi + pad)
            ax.legend(loc="best", fontsize=8)
    run = csv.split("/outputs/")[-1].split("/metrics/")[0]
    fig.suptitle(f"step {last_step} | {run}", y=1.0)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=90)
    plt.close(fig)
    return last_step


while True:
    try:
        csv = newest_csv()
        if csv:
            step = plot(csv)
            print(f"[{time.strftime('%H:%M:%S')}] updated {OUT_PNG} | step={step} | {csv}", flush=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] no metrics.csv found", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] error: {e}", flush=True)
    time.sleep(INTERVAL)
