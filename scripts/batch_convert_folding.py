"""Batch-convert all downloaded qcez folding episodes (LeRobot v2.1 parquet) into the
iws HDF5 layout for the bimanual_fold EEF action mode.

Split: first N_TRAIN episodes -> data/folding/train, the rest -> data/folding/val.
Original episode indices are preserved in filenames for traceability.
"""
import os, io, glob, shutil
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import cv2
import h5py

REPO = "/media/skr/storage/YC/interactive_world_sim"
RAW = f"{REPO}/data/folding_raw/data/chunk-000"
OUT = f"{REPO}/data/folding"
CAM = "observation.images.cam_high"
RES = 128
N_TRAIN = 45  # remaining -> val


def decode(cell):
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


def convert(parquet, out_hdf5):
    df = pq.read_table(parquet).to_pandas()
    joints = np.stack(df["observation.state"].to_numpy()).astype(np.float32)  # (T,14) -> FK -> EEF
    action = np.stack(df["action"].to_numpy()).astype(np.float32)             # (T,14) length only
    T = len(df)
    small = np.stack([cv2.resize(decode(c), (RES, RES), interpolation=cv2.INTER_AREA)
                      for c in df[CAM].to_numpy()]).astype(np.uint8)
    base = np.tile(np.eye(4, dtype=np.float32), (T, 2, 1, 1))
    os.makedirs(os.path.dirname(out_hdf5), exist_ok=True)
    with h5py.File(out_hdf5, "w") as f:
        f.create_dataset("action", data=action)
        obs = f.create_group("obs")
        obs.create_dataset("joint_pos", data=joints)
        obs.create_dataset("full_joint_pos", data=joints)
        obs.create_dataset("world_t_robot_base", data=base)
        obs.create_group("images").create_dataset(
            "cam_high", data=small, chunks=(1, RES, RES, 3))
    return T


def main():
    # fresh start
    for sub in ("train", "val"):
        d = os.path.join(OUT, sub)
        if os.path.isdir(d):
            shutil.rmtree(d)
    for f in glob.glob(os.path.join(OUT, "train", "*.zarr.zip")) + glob.glob(os.path.join(OUT, "val", "*.zarr.zip")):
        os.remove(f)

    files = sorted(glob.glob(os.path.join(RAW, "episode_*.parquet")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    print(f"found {len(files)} episodes; split {N_TRAIN} train / {len(files)-N_TRAIN} val")
    tot_train = tot_val = 0
    for i, fp in enumerate(files):
        idx = int(fp.split("_")[-1].split(".")[0])
        split = "train" if i < N_TRAIN else "val"
        out = os.path.join(OUT, split, f"episode_{idx}.hdf5")
        T = convert(fp, out)
        if split == "train":
            tot_train += T
        else:
            tot_val += T
        print(f"  [{i+1:2d}/{len(files)}] {split:5s} episode_{idx}.hdf5  T={T}")
    print(f"\nDONE. train frames={tot_train}  val frames={tot_val}")


if __name__ == "__main__":
    main()
