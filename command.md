# Training commands — folding world model (200 eps, EEF `bimanual_fold`)

Env: `/media/skr/storage/conda_envs/iws/bin/python` (cu128, Blackwell)
Data: `data/folding` (180 train / 20 val, 50 fps HDF5)
Action: EEF `bimanual_fold`, `action_dim=8` · Camera: `cam_high` · Res: 128 · Batch: 4 (16 OOMs)
Runs are launched **detached** (`setsid nohup … &`) so they survive session teardown.

---

## Stage 1 — Autoencoder (encoder + decoder)
`skip_frame=1` (uses all frames). Ceiling 80k steps; stop when val reconstruction plateaus.

```bash
cd /media/skr/storage/YC/interactive_world_sim
mkdir -p outputs/stage1_200
export HF_HUB_ENABLE_HF_TRANSFER=0
setsid nohup env WANDB_MODE=disabled /media/skr/storage/conda_envs/iws/bin/python main.py \
  +name=folding_stage1_200 \
  algorithm=latent_world_model experiment=exp_latent_dyn dataset=folding_dataset \
  dataset.dataset_dir=data/folding \
  dataset.horizon=1 dataset.val_horizon=1 dataset.skip_frame=1 \
  dataset.obs_keys=[cam_high] dataset.action_mode=bimanual_fold \
  experiment.training.batch_size=4 experiment.training.max_steps=80000 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.val_every_n_step=3000 experiment.validation.batch_size=4 experiment.validation.limit_batch=1 \
  experiment.training.data.num_workers=4 experiment.validation.data.num_workers=4 \
  experiment.training.checkpointing.every_n_train_steps=3000 +experiment.training.checkpointing.save_last=true \
  algorithm.latent_dim=512 algorithm.action_dim=8 algorithm.training_stage=1 \
  wandb.mode=disabled wandb.entity=none > outputs/stage1_200/train.log 2>&1 < /dev/null &
```

---

## Stage 2 — Latent dynamics (action-conditioned)
`skip_frame=5` + option D (action subsampled to 8-dim). Needs the Stage-1 checkpoint.
Set `CKPT` to the best/last Stage-1 checkpoint.

```bash
cd /media/skr/storage/YC/interactive_world_sim
mkdir -p outputs/stage2_200
export HF_HUB_ENABLE_HF_TRANSFER=0
CKPT=outputs/<STAGE1_RUN_DIR>/checkpoints/last.ckpt      # <-- edit
setsid nohup env WANDB_MODE=disabled /media/skr/storage/conda_envs/iws/bin/python main.py \
  +name=folding_stage2_200 \
  algorithm=latent_world_model experiment=exp_latent_dyn dataset=folding_dataset \
  dataset.dataset_dir=data/folding \
  dataset.horizon=10 dataset.val_horizon=200 dataset.skip_frame=5 \
  dataset.obs_keys=[cam_high] dataset.action_mode=bimanual_fold \
  experiment.training.batch_size=4 experiment.training.max_steps=200000 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.val_every_n_step=5000 experiment.validation.batch_size=2 experiment.validation.limit_batch=1 \
  experiment.training.data.num_workers=4 experiment.validation.data.num_workers=4 \
  experiment.training.checkpointing.every_n_train_steps=5000 +experiment.training.checkpointing.save_last=true \
  algorithm.latent_dim=512 algorithm.action_dim=8 \
  algorithm.noise_scheduler.loss_weighting=uniform algorithm.sampling_strategy=terminal_only \
  algorithm.load_ae="$CKPT" algorithm.training_stage=2 \
  wandb.mode=disabled wandb.entity=none > outputs/stage2_200/train.log 2>&1 < /dev/null &
```

---

## Stage 3 — Decoder finetune (robust to noisy latents)
`horizon=1`. Needs the Stage-2 checkpoint. (`skip_frame` here barely matters; `=1` keeps most frames.)

```bash
cd /media/skr/storage/YC/interactive_world_sim
mkdir -p outputs/stage3_200
export HF_HUB_ENABLE_HF_TRANSFER=0
CKPT=outputs/<STAGE2_RUN_DIR>/checkpoints/last.ckpt      # <-- edit
setsid nohup env WANDB_MODE=disabled /media/skr/storage/conda_envs/iws/bin/python main.py \
  +name=folding_stage3_200 \
  algorithm=latent_world_model experiment=exp_latent_dyn dataset=folding_dataset \
  dataset.dataset_dir=data/folding \
  dataset.horizon=1 dataset.val_horizon=200 dataset.skip_frame=1 \
  dataset.obs_keys=[cam_high] dataset.action_mode=bimanual_fold \
  experiment.training.batch_size=4 experiment.training.max_steps=60000 \
  experiment.training.log_every_n_steps=100 \
  experiment.validation.val_every_n_step=5000 experiment.validation.batch_size=2 experiment.validation.limit_batch=1 \
  experiment.training.data.num_workers=4 experiment.validation.data.num_workers=4 \
  experiment.training.checkpointing.every_n_train_steps=5000 +experiment.training.checkpointing.save_last=true \
  algorithm.latent_dim=512 algorithm.action_dim=8 \
  algorithm.noise_scheduler.loss_weighting=uniform algorithm.sampling_strategy=terminal_only \
  algorithm.load_ae="$CKPT" algorithm.training_stage=3 \
  wandb.mode=disabled wandb.entity=none > outputs/stage3_200/train.log 2>&1 < /dev/null &
```

---

## Monitoring

```bash
# find the run dir
grep -m1 "Outputs will be saved" outputs/stage1_200/train.log

# GT-vs-reconstruction from latest checkpoint (Stage 1)
python scripts/render_recon.py outputs/<RUN_DIR> [dec_steps] [n_frames]

# clip-bounds re-check (if data changes)
python scripts/check_clip_bounds.py

# live loss plot (detached, refresh 300s) -> outputs/latest-run/loss_curves.png
setsid nohup /media/skr/storage/conda_envs/iws/bin/python scripts/live_loss_plot.py 300 \
  > outputs/latest-run/live_plot.log 2>&1 < /dev/null &

# GPU / process
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
pgrep -af folding_stage1_200
```

## Notes
- **Stop criterion:** watch `validation/mse` ↓ / `validation/psnr` ↑ in the live plot; stop each stage when it plateaus (no built-in early stopping).
- **Order:** Stage 1 → 2 → 3, each `load_ae` = previous stage's checkpoint.
- **Decisions:** D1 (EEF `bimanual_fold`, 8-dim) and D2 (subsample action, `skip_frame=5` for Stage 2/3) — see `PIPELINE_PLAN.md`.
