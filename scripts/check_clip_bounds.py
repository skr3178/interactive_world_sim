"""Validate the bimanual_fold EEF clip box against the actual data.

Runs FK over all downloaded LeRobot folding parquets and reports, per axis:
  - data min/max, the current clip box, and the % of EEF samples that get clipped.
A proper box clips ~0% of real data. If anything clips, widen the box (suggested
bounds = data range + margin are printed). Re-run after downloading more episodes.

Usage:
  python scripts/check_clip_bounds.py
"""
import glob, argparse
import numpy as np
import pyarrow.parquet as pq
from yixuan_utilities.kinematics_helper import KinHelper

# keep in sync with ctrl_mode="bimanual_fold" in utils/action_utils.py
BOX = dict(x=(0.23, 0.68), y=(-0.45, 0.50), z=(-0.47, 0.18))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/folding_raw/data/chunk-000/episode_*.parquet")
    ap.add_argument("--stride", type=int, default=10, help="frame subsample for speed")
    ap.add_argument("--margin", type=float, default=0.05)
    args = ap.parse_args()

    kin = KinHelper("trossen_vx300s")
    files = sorted(glob.glob(args.glob))
    print(f"episodes: {len(files)}")
    allp, per_ep = [], []
    for fp in files:
        st = np.stack(pq.read_table(fp).to_pandas()["observation.state"].to_numpy()).astype(np.float64)
        pts = []
        for t in range(0, len(st), args.stride):
            for r in range(2):
                fk = np.concatenate([st[t][r*7:(r+1)*7], st[t][r*7+6:r*7+7]])
                pts.append(kin.compute_fk_from_link_idx(fk, [kin.sapien_eef_idx])[0][:3, 3])
        pts = np.array(pts); allp.append(pts)
        out = sum(((pts[:, i] < BOX[ax][0]) | (pts[:, i] > BOX[ax][1])).sum()
                  for i, ax in enumerate("xyz"))
        if out:
            per_ep.append((fp.split("_")[-1].split(".")[0], int(out), len(pts)))
    allp = np.concatenate(allp, 0)
    print(f"total EEF samples: {len(allp)}\n{'axis':4}{'min':>9}{'max':>9}{'box_lo':>8}{'box_hi':>8}{'clip%':>8}")
    ok = True
    for i, ax in enumerate("xyz"):
        lo, hi = BOX[ax]; v = allp[:, i]
        clp = ((v < lo) | (v > hi)).mean() * 100
        ok = ok and clp == 0.0
        print(f"{ax:4}{v.min():9.3f}{v.max():9.3f}{lo:8.2f}{hi:8.2f}{clp:7.2f}%")
    print("\nepisodes with out-of-box samples:", per_ep or "none")
    print("suggested box (data range + margin):",
          {ax: (round(allp[:, i].min()-args.margin, 2), round(allp[:, i].max()+args.margin, 2))
           for i, ax in enumerate("xyz")})
    print("\nRESULT:", "OK — box encompasses all data" if ok else "WIDEN — some data is being clipped")

if __name__ == "__main__":
    main()
