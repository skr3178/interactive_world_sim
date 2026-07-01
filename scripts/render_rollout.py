"""Render a Stage-2 predicted-vs-GT rollout as an MP4.

Replicates validation_step (training_stage=2): encode start frame -> autoregressively roll the
dynamics forward with the real action sequence -> decode predicted latents -> frames.
Saves a side-by-side video (left = GT, right = predicted).

Usage: python scripts/render_rollout.py <run_dir> [ckpt] [val_episode_idx] [max_frames]
"""
import sys, glob
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from einops import rearrange
import imageio

from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
from interactive_world_sim.datasets.latent_dynamics.real_aloha_dataset import RealAlohaDataset
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

OmegaConf.register_new_resolver("eval", lambda e: eval(e, {"np": np}))
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))

run_dir = sys.argv[1].rstrip("/")
ckpt = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob(f"{run_dir}/checkpoints/*.ckpt"))[-1]
ep_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 200
device = sys.argv[5] if len(sys.argv) > 5 else "cuda"   # 'cpu' to avoid touching a training GPU
roll_len = int(sys.argv[6]) if len(sys.argv) > 6 else None  # override val_horizon (shorter = faster on CPU)

cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")
if roll_len:
    cfg.dataset.val_horizon = roll_len
model = LatentWorldModel(cfg.algorithm)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(sd["state_dict"], strict=False)
model.eval().to(device)
print(f"loaded {ckpt} | stage {model.training_stage} | missing={len(missing)}")

val = RealAlohaDataset(cfg.dataset).get_validation_dataset()
s = val[ep_idx]
batch = {
    "obs": {k: torch.as_tensor(s["obs"][k]).unsqueeze(0).to(device).float() for k in model.obs_keys},
    "action": torch.as_tensor(s["action"]).unsqueeze(0).to(device).float(),
}

with torch.no_grad():
    obs = torch.cat([model.normalizer[k].normalize(batch["obs"][k]) for k in model.obs_keys], dim=2).float()
    action = model.normalizer["action"].normalize(batch["action"]).float()
    xs = rearrange(obs, "b t c h w -> (b t) c h w")
    z_gt = model.encoder_forward(xs)
    z_gt = rearrange(z_gt, "(b t) c h w -> b t c h w", b=obs.shape[0])

    # --- training_stage==2 autoregressive rollout (copied from validation_step) ---
    z_0 = z_gt[:, 0]
    z_seq_ls, z_last, horizon = [], z_0.clone(), z_gt.shape[1]
    for i in range(1, action.shape[1], horizon):
        ac = action[:, i : i + horizon]
        sz = ac.shape[1]
        if sz < horizon:
            ac = F.pad(ac, (0, 0, 0, horizon - sz), mode="replicate")
        zs = model.dynamics_forward(z_last[:, None], ac)[:, :sz]
        z_seq_ls.append(zs)
        z_last = zs[:, -1].clone()
    z_seq = torch.cat([z_0.unsqueeze(1), torch.cat(z_seq_ls, 1)], 1)
    z_seq = rearrange(z_seq, "b t c h w -> (b t) c h w")

    xs_pred = render_img_cm(model, z_seq, xs.shape[-1], model.normalizer, num_views=model.num_views)
    xs_pred = rearrange(xs_pred, "(b t) c h w -> t b c h w", b=obs.shape[0])[:, 0]
    xs_gt = rearrange(torch.cat([batch["obs"][k] for k in model.obs_keys], dim=2),
                      "b t c h w -> t b c h w", b=obs.shape[0])[:, 0]

pred = xs_pred.detach().cpu().clamp(0, 1).numpy()
gt = xs_gt.detach().cpu().clamp(0, 1).numpy()
T = min(len(pred), len(gt), max_frames)
mse = float(((pred[:T] - gt[:T]) ** 2).mean())
print(f"rollout frames: {T} | rollout MSE (pixel): {mse:.5f}")

def to_img(a):
    return (a.transpose(1, 2, 0) * 255).astype(np.uint8)

out = f"{run_dir}/rollout.mp4"
sep = np.full((gt.shape[2], 4, 3), 255, np.uint8)
w = imageio.get_writer(out, fps=10, codec="libx264", macro_block_size=16,
                       ffmpeg_params=["-pix_fmt", "yuv420p"])
for t in range(T):
    w.append_data(np.concatenate([to_img(gt[t]), sep, to_img(pred[t])], axis=1))  # left=GT, right=pred
w.close()
print(f"wrote {out}  (left = GT, right = predicted, {T} frames @ 10fps = {T/10:.0f}s)")
