"""Convert a LeRobot v2.1 ALOHA folding episode (parquet, images embedded as PNG bytes)
into the iws HDF5 layout, using ctrl_mode='joint' (action = 14-dim joints, passthrough).

Sample run: one episode -> data/folding/train/episode_0.hdf5  (+ a viewing MP4).
"""
import os, io, json, argparse
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import cv2
import h5py
import imageio

REPO = "/media/skr/storage/YC/interactive_world_sim"

def decode_img(cell):
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))  # (H,W,3) uint8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=f"{REPO}/data/folding_raw/data/chunk-000/episode_000000.parquet")
    ap.add_argument("--cam", default="observation.images.cam_high")
    ap.add_argument("--out_hdf5", default=f"{REPO}/data/folding/train/episode_0.hdf5")
    ap.add_argument("--out_mp4", default=f"{REPO}/data/folding/preview/episode_0_cam_high.mp4")
    ap.add_argument("--res", type=int, default=128)   # training resolution stored in HDF5
    ap.add_argument("--skip", type=int, default=1)    # 1 = store all frames; subsample at load via skip_frame
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()

    df = pq.read_table(args.parquet).to_pandas()
    sel = slice(None, None, args.skip)
    df = df.iloc[sel].reset_index(drop=True)
    T = len(df)
    print(f"episode rows (after skip={args.skip}): T={T}")

    # For the EEF (bimanual_fold) action_mode the loader runs FK on obs/joint_pos,
    # so joint_pos must be the *measured* joint configuration that matches the images.
    # observation.state = measured follower joints; action = commanded targets.
    joints = np.stack(df["observation.state"].to_numpy()).astype(np.float32)  # (T,14)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)             # (T,14) cmd (length only)
    assert joints.shape[1] % 7 == 0, joints.shape
    print("joints(state) shape:", joints.shape, "dims % 7 ==", joints.shape[1] % 7)

    # decode + (native) frames for viewing, resized frames for HDF5
    native, small = [], []
    for cell in df[args.cam].to_numpy():
        im = decode_img(cell)
        native.append(im)
        small.append(cv2.resize(im, (args.res, args.res), interpolation=cv2.INTER_AREA))
    native = np.stack(native)            # (T,224,224,3)
    small = np.stack(small).astype(np.uint8)   # (T,res,res,3)
    print("native frames:", native.shape, "| stored frames:", small.shape)

    # --- write viewing MP4 (native res) ---
    os.makedirs(os.path.dirname(args.out_mp4), exist_ok=True)
    w = imageio.get_writer(args.out_mp4, fps=args.fps, codec="libx264",
                           macro_block_size=16, ffmpeg_params=["-pix_fmt", "yuv420p"])
    for im in native:
        w.append_data(im)
    w.close()
    print("wrote MP4:", args.out_mp4)

    # --- write iws HDF5 (mimic ALOHA layout; ctrl_mode='joint' will pass joint_pos through) ---
    os.makedirs(os.path.dirname(args.out_hdf5), exist_ok=True)
    base = np.tile(np.eye(4, dtype=np.float32), (T, 2, 1, 1))  # (T,2,4,4) stub, unused in joint mode
    with h5py.File(args.out_hdf5, "w") as f:
        f.create_dataset("action", data=action)                       # length only
        obs = f.create_group("obs")
        obs.create_dataset("joint_pos", data=joints)                  # measured joints -> FK -> EEF action
        obs.create_dataset("full_joint_pos", data=joints)             # length only
        obs.create_dataset("world_t_robot_base", data=base)           # identity base -> EEF in robot frame
        imgs = obs.create_group("images")
        imgs.create_dataset("cam_high", data=small, chunks=(1, args.res, args.res, 3))
    sz = os.path.getsize(args.out_hdf5) / 1e6
    print(f"wrote HDF5: {args.out_hdf5}  ({sz:.1f} MB)")
    print("HDF5 keys: action, obs/joint_pos, obs/full_joint_pos, obs/world_t_robot_base, obs/images/cam_high")

if __name__ == "__main__":
    main()
