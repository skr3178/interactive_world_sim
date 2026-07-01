"""Render a GT-vs-reconstruction grid from a Stage-1 checkpoint.

Replicates the autoencoder path of validation_step: normalize obs -> encoder_forward
-> render_img_cm (consistency-model decode) -> unnormalize. Saves a 2-row image:
top = ground-truth frames, bottom = encode->decode reconstructions.

Usage: python scripts/render_recon.py <run_dir>
  e.g. python scripts/render_recon.py outputs/2026-06-30/20-10-00
"""
import sys, glob
import numpy as np
import torch
from omegaconf import OmegaConf
from einops import rearrange
import imageio

from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
from interactive_world_sim.datasets.latent_dynamics.real_aloha_dataset import RealAlohaDataset
from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

# custom resolvers used in the saved hydra config (mirrors main.py)
OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))

run_dir = sys.argv[1].rstrip("/")
dec_steps = int(sys.argv[2]) if len(sys.argv) > 2 else None   # override decode steps
n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ckpts = sorted(glob.glob(f"{run_dir}/checkpoints/*.ckpt"))
assert ckpts, f"no checkpoint in {run_dir}/checkpoints"
ckpt = ckpts[-1]
print("checkpoint:", ckpt)

cfg = OmegaConf.load(f"{run_dir}/.hydra/config.yaml")
device = "cuda"

model = LatentWorldModel(cfg.algorithm)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(sd["state_dict"], strict=False)
print(f"loaded ckpt (missing={len(missing)} unexpected={len(unexpected)})")
if dec_steps is not None:
    model.dec_infer_steps = dec_steps
print("dec_infer_steps:", model.dec_infer_steps)
model.eval().to(device)

ds = RealAlohaDataset(cfg.dataset)
idxs = np.linspace(0, len(ds) - 1, n_frames).astype(int)
obs = torch.stack([torch.as_tensor(ds[i]["obs"]["cam_high"]) for i in idxs], 0).to(device).float()
print("obs batch:", tuple(obs.shape), "range", float(obs.min()), float(obs.max()))

with torch.no_grad():
    obs_n = model.normalizer["cam_high"].normalize(obs)        # (B,1,C,H,W)
    xs = rearrange(obs_n, "b t c h w -> (b t) c h w")
    z = model.encoder_forward(xs)
    xs_pred = render_img_cm(model, z, xs.shape[-1], model.normalizer, num_views=1)

gt = rearrange(obs, "b t c h w -> (b t) c h w").detach().cpu().clamp(0, 1)
pred = xs_pred.detach().cpu().float().clamp(0, 1)


def to_img(t):
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


top = np.concatenate([to_img(gt[i]) for i in range(len(idxs))], axis=1)
bot = np.concatenate([to_img(pred[i]) for i in range(len(idxs))], axis=1)
sep = np.full((4, top.shape[1], 3), 255, np.uint8)
grid = np.concatenate([top, sep, bot], axis=0)
out = f"{run_dir}/recon_steps{model.dec_infer_steps}.png"
imageio.imwrite(out, grid)
mse = torch.mean((gt - pred) ** 2).item()
print(f"recon MSE (display scale): {mse:.5f}")
print("wrote", out, "| top=GT  bottom=reconstruction")
