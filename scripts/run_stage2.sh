#!/bin/bash
# Stage 2 — latent dynamics (skip_frame=5, option D, 8-dim EEF @ 10fps)
# Built on the 200-ep Stage-1 autoencoder (step 15000, near-perfect).
cd /media/skr/storage/YC/interactive_world_sim
export HF_HUB_ENABLE_HF_TRANSFER=0
export WANDB_MODE=disabled
mkdir -p outputs/stage2_200
CKPT=outputs/2026-07-01/12-54-48/checkpoints/stage1_final.ckpt

# Auto-resume: if a prior Stage-2 checkpoint exists, continue from it (full state: step +
# optimizer via trainer.fit ckpt_path) instead of restarting the dynamics from 0. Picks the
# most-recent Stage-2 run's last.ckpt (no '=' in the name -> hydra-safe).
RESUME_ARG=""
for c in $(ls -t outputs/*/*/checkpoints/last.ckpt 2>/dev/null); do
  d=$(dirname "$(dirname "$c")")
  if grep -q "training_stage=2" "$d/.hydra/overrides.yaml" 2>/dev/null; then
    RESUME_ARG="load=$c"; echo ">>> RESUMING Stage 2 from: $c"; break
  fi
done
[ -z "$RESUME_ARG" ] && echo ">>> No prior Stage-2 checkpoint — starting dynamics fresh."

/media/skr/storage/conda_envs/iws/bin/python main.py \
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
  $RESUME_ARG \
  wandb.mode=disabled wandb.entity=none 2>&1 | tee outputs/stage2_200/train.log
