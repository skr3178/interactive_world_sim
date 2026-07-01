"""Live loss-curve plotter. Every INTERVAL seconds, finds the most recently-written
offline wandb run under outputs/, extracts scalar history, and regenerates
outputs/latest-run/loss_curves.png. Runs until killed.
"""
import os, sys, glob, json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

REPO = "/media/skr/storage/YC/interactive_world_sim"
OUT_DIR = f"{REPO}/outputs/latest-run"
OUT_PNG = f"{OUT_DIR}/loss_curves.png"
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 300
os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ("training/rec_loss", "validation/mse", "validation/psnr",
           "validation/ssim", "validation/uiqi")

# near-perfect reconstruction targets (the gate before Stage 2)
TARGETS = {
    "training/rec_loss": 0.001,
    "validation/mse": 0.001,   # ~PSNR 30
    "validation/psnr": 30.0,   # dB
    "validation/ssim": 0.90,
    "validation/uiqi": 0.90,
}


def newest_wandb():
    files = glob.glob(f"{REPO}/outputs/**/offline-run-*/*.wandb", recursive=True)
    return max(files, key=os.path.getmtime) if files else None


def read_history(wf):
    ds = DataStore()
    ds.open_for_scan(wf)
    rows = []
    while True:
        try:
            d = ds.scan_data()
        except Exception:
            break
        if d is None:
            break
        r = pb.Record()
        r.ParseFromString(d)
        if r.WhichOneof("record_type") == "history":
            item = {}
            for h in r.history.item:
                k = h.key or "/".join(h.nested_key)
                try:
                    item[k] = json.loads(h.value_json)
                except Exception:
                    item[k] = h.value_json
            rows.append(item)
    return rows


def plot(rows, wf):
    sk = "trainer/global_step"
    present = [m for m in METRICS if any(m in r for r in rows)]
    if not present:
        return None
    last_step = max((r.get(sk, 0) for r in rows), default=0)
    fig = plt.figure(figsize=(11, 2.6 * len(present)))
    for i, mk in enumerate(present, 1):
        xs = [r[sk] for r in rows if mk in r and sk in r]
        ys = [r[mk] for r in rows if mk in r and sk in r]
        ax = plt.subplot(len(present), 1, i)
        ax.plot(xs, ys, lw=1, marker=".", ms=4)
        ax.set_title(f"{mk}  (last={ys[-1]:.4f})" if ys else mk)
        ax.set_xlabel(sk)
        ax.grid(alpha=0.3)
        if mk in TARGETS and ys:
            tgt = TARGETS[mk]
            ax.axhline(tgt, color="red", ls=":", lw=1.3, label=f"target {tgt}")
            lo, hi = min(min(ys), tgt), max(max(ys), tgt)
            pad = (hi - lo) * 0.08 or 0.001
            ax.set_ylim(lo - pad, hi + pad)
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"step {last_step} | {os.path.basename(os.path.dirname(os.path.dirname(wf)))}", y=1.0)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=90)
    plt.close(fig)
    return last_step


while True:
    try:
        wf = newest_wandb()
        if wf:
            rows = read_history(wf)
            step = plot(rows, wf)
            print(f"[{time.strftime('%H:%M:%S')}] updated {OUT_PNG} | rows={len(rows)} step={step}", flush=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] no wandb run found", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] error: {e}", flush=True)
    time.sleep(INTERVAL)
