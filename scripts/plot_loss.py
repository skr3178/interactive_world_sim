"""Extract scalar history from an offline wandb run and plot loss curves."""
import sys, json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

run_dir = sys.argv[1].rstrip("/")
wf = glob.glob(f"{run_dir}/wandb/offline-run-*/*.wandb")[0]
print("reading", wf)

ds = DataStore()
ds.open_for_scan(wf)
rows = []
while True:
    try:
        data = ds.scan_data()
    except Exception:
        break
    if data is None:
        break
    rec = pb.Record()
    rec.ParseFromString(data)
    if rec.WhichOneof("record_type") == "history":
        item = {}
        for h in rec.history.item:
            key = h.key or "/".join(h.nested_key)
            try:
                item[key] = json.loads(h.value_json)
            except Exception:
                item[key] = h.value_json
        rows.append(item)

print("history rows:", len(rows))
allkeys = sorted(set().union(*[set(r.keys()) for r in rows])) if rows else []
print("keys:", allkeys)

# pick a step axis
step_key = next((k for k in ("trainer/global_step", "_step", "global_step") if k in allkeys), None)
metric_keys = [k for k in allkeys if any(t in k.lower() for t in ("loss", "fvd", "lpips", "fid"))
               or k in ("validation/mse", "validation/psnr", "validation/ssim", "validation/uiqi")]
print("step_key:", step_key, "| metrics:", metric_keys)

if step_key and metric_keys:
    plt.figure(figsize=(11, 4 * len(metric_keys)))
    for i, mk in enumerate(metric_keys, 1):
        xs = [r[step_key] for r in rows if mk in r and step_key in r]
        ys = [r[mk] for r in rows if mk in r and step_key in r]
        if not xs:
            continue
        ax = plt.subplot(len(metric_keys), 1, i)
        ax.plot(xs, ys, lw=1)
        ax.set_title(mk)
        ax.set_xlabel(step_key)
        ax.grid(alpha=0.3)
        print(f"  {mk}: {len(xs)} pts, first={ys[0]:.4f} last={ys[-1]:.4f} min={min(ys):.4f}")
    plt.tight_layout()
    out = f"{run_dir}/loss_curves.png"
    plt.savefig(out, dpi=90)
    print("wrote", out)
else:
    print("no plottable step/metric keys found")
