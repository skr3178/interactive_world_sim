# Interactive World Sim — Cloth-Folding Pipeline Plan

Plan for training the `interactive_world_sim` (iws) learned world model on an external
clothes-folding dataset and running an interactive (keyboard / replay) demo.

Repo: `/media/skr/storage/YC/interactive_world_sim`
Paper: *Interactive World Simulator for Robot Policy Training and Evaluation* (Wang et al.)
Env: conda `iws` at `/media/skr/storage/conda_envs/iws` (cu128 build for Blackwell sm_120)

---

## 1. Goal

Two possible objectives (decide scope before starting):

- **Controllability demo** — keyboard / replay-driven action-conditioned rollouts of folding.
  Achievable, good showcase. *This is the realistic first target.*
- **Faithful policy evaluator** — closed-loop, in-sim success correlates with real-robot success.
  Much harder; needs play-data coverage we don't have. Stretch goal.

The single number that proves the eval claim (if pursued later): **rank correlation between
in-world-model success rate and real-robot success rate on held-out policies.**

---

## 2. Dataset decision

### Selected: **`qcez/folding_clothes_200_1_14_lerobot`** (Option A)

| Property | Value |
|---|---|
| Robot | **ALOHA** (same family as the repo's native data) |
| Format | LeRobot **v2.1**, images stored as frames (no video decode) |
| Volume | 200 episodes / 245,348 frames @ 50 fps |
| Action | `float32 [14]` (2 arms × 7 = 6 joints + gripper) |
| State | `float32 [14]` |
| Cameras | 3 × `[3, 224, 224]`: `cam_high`, `cam_left_wrist`, `cam_right_wrist` |
| Matched policy | ✅ `qcez/fold-cloth-1_14-200-checkpoint` |

**Why A over the alternatives:**
- ALOHA = repo's native robot → 14-dim action maps onto existing `bimanual_box` path (least porting).
- Images-as-frames → converter is just "read → write HDF5" (no AV1/h264 decode, no v3.0 unpacking).
- Matched policy exists → enables the **full eval loop** later (train WM → drop policy in → compare).

### Alternative: `qcez/folding_clothes_250_lerobot_av1` (Option B)
Franka bimanual, v3.0, 250 eps / 430k frames @ 20 fps, action `[14]`, 3 cams (480×640) **video**.
More raw data, but costs: video decode + v3.0 multi-episode packing + non-native robot + no matched policy.
Only choose B if maximizing data volume outweighs the extra converter work.

### Key insight about qcez
qcez's **models are policies** (`pi0`, ACT/LoRA, `value_function`), not world models. They ship
**matched dataset ↔ policy pairs** — valuable because the policy becomes the thing you evaluate
*inside* your trained world model (the actual paper / Fern pitch).

---

## 3. Data-format facts (established)

**What the iws model consumes:** per timestep `[RGB frame(s) (3,128,128) normalized] + [action vector (action_dim,)]`.
Sequence of these = an episode. No depth/proprioception fed to the model.

**iws native HDF5 (ALOHA) per-episode layout** (target format):
- `obs/images/camera_X_color` `(T,H,W,3)` uint8
- `action` derived from `obs/joint_pos` via FK (`KinHelper("trossen_vx300s")`) — **we bypass this**
- also stores `obs/ee_pos`, `obs/world_t_robot_base`, intrinsics/extrinsics (not needed by model)

**Camera count is a config knob**, not fixed at 2: architecture derives everything from
`len(obs_keys)` (`num_views`, `x_shape = 3*num_views`, `num_latent_channel = 4*num_views`).
Paper's released checkpoints each use **one** camera. → Use `cam_high` only; ignore wrist cams
(they move with the arm = much harder to model, and cost more channels/data).

---

## 4. The converter (the main work)

### A. LeRobot v2.1 → iws HDF5 converter
Per episode:
- split parquet by `episode_index`
- read frames (`cam_high`) → `obs/images/camera_0_color` `(T,H,W,3)` uint8
- read `action` column → `action` `(T,14)` float32
- apply `skip_frame ≈ 5` (50 fps → ~10 fps effective, so there's real motion per step)
- write to `data/folding/{train,val}/episode_N.hdf5` (e.g. 180 train / 20 val)

### B. Small loader branch (avoid ALOHA FK)
`_convert_real_to_dp_replay` in
`interactive_world_sim/datasets/latent_dynamics/real_aloha_dataset.py`
is hardwired to ALOHA keys (`full_joint_pos`, `joint_pos`, `world_t_robot_base`) + FK.
Add an `action_mode="joint_direct"` path (~30 lines) that reads `action` straight from the
HDF5 and skips `joint_pos_to_action_primitive` / `KinHelper`.

(Alternative hack: store the 14-dim action as `obs/joint_pos` + dummy `world_t_robot_base`
and reuse existing `ctrl_mode="joint"`. Messier; prefer the clean branch.)

### Config
- `action_mode=bimanual_fold`, `action_dim=8`  *(EEF mode — see Decision log D1)*
- `obs_keys=[cam_high]`, `resolution=128` (downsample from 224)
- `skip_frame=5`

Converter stores `observation.state` (measured joints) as `obs/joint_pos`; the loader runs
FK on it to produce the 8-dim EEF action. `world_t_robot_base` = identity stub → EEF in robot frame.

---

## Decision log

### D1 — Action representation: **EEF (`bimanual_fold`)**, not raw joints  *(2026-06-30)*
**Fork:** condition the world model on raw 14-dim joints (`ctrl_mode=joint`, zero-code) vs. a
forward-kinematics **end-effector** action (per-arm world-frame XYZ + gripper, 8-dim).

**Choice: EEF.** **Why:** the final UI needs **end-effector control** (WASD = move the gripper).
That only works natively if the model is *trained* in EEF space — exactly how the released
`bimanual_rope` model gets its keyboard EEF control. A joint-trained model would need an IK
layer bolted onto the UI (eef→joints every step, off-distribution risk). The action
representation is baked in at training time, so this had to be decided before Stage 1.

**Implementation:**
- New `ctrl_mode="bimanual_fold"` in `utils/action_utils.py` — copy of `bimanual_rope`
  (FK via `KinHelper("trossen_vx300s")`, 8-dim per-arm XYZ+gripper) with a folding workspace box.
- Converter feeds **`observation.state`** (measured joints, matches the images) into `obs/joint_pos`.
- Reusing the same robot (ALOHA) means FK + gripper handling work unchanged.

**Why not reuse `bimanual_rope` verbatim — the clip box.** The rope mode hardcodes a workspace
clip box; the clips exist to (a) bound the action space to the task region (easier to learn,
keeps control on-distribution) and (b) guard real-robot IK. They were hand-tuned per task from
that task's play-data EEF range. Measured FK on our folding data showed it falls **entirely
outside** the rope box — especially **z**:

| axis | rope box | our folding EEF (ep0, robot frame) |
|---|---|---|
| x | [0.20, 0.40] | [0.33, 0.61] |
| y | [−0.15, 0.15] | [−0.35, 0.40] |
| **z** | **[0.08, 0.16]** | **[−0.42, −0.04]** (rope box would collapse all z → 0.08) |

So `bimanual_fold` uses a box = **our measured range + margin**.

**LOCKED clip box — validated over the FULL 200 episodes:**

| axis | lower | upper | data range (200 ep) | clip% |
|---|---|---|---|---|
| x | **0.23** | **0.68** | [0.279, 0.630] | 0.00% |
| y | **−0.45** | **0.50** | [−0.403, 0.466] | 0.00% |
| z | **−0.47** | **0.18** | [−0.418, 0.088] | 0.00% |

Validated by `scripts/check_clip_bounds.py` over all **200 episodes (49,244 EEF samples)**:
**0.00% clipped, "box encompasses all data," episodes out of box: none.** Why this is final-enough
for training: a clip box only distorts data if it's too *tight*; too-loose is harmless, and **action
normalization uses the real data stats, not the box**, so looseness costs nothing. Tight bounds only
matter later for the keyboard control guardrail.

**Resolved (2026-07-01) — no widening needed after downloading all 200:** the box was originally
locked on the 50-ep working set with a warning that the remaining 150 might push `z-upper` / `x-lower`
and force a re-scan + widen. The full-200 rescan **disproved that**: the real EEF envelope stays well
inside the box on exactly those axes (z tops out at 0.088 vs box 0.18; x floors at 0.279 vs box 0.23).
The 50-ep box had enough margin that the extra 150 episodes added no new extremes, so the box is
unchanged and now validated against the entire dataset. (The script's suggested data-range+margin box
would actually *tighten* z-upper to 0.14 — pointless since clip% is already 0.) Note: dataset/URDF
limits can't replace this scan — physical joint limits are far too loose for the box, and the JSON's
recorded limits are joint-space (our action is EEF), so the FK scan is the correct "limits from the
dataset."

**Method note (how limits were determined):** FK over actual EEF trajectories, not robot physical
limits (too loose) nor raw joint stats (wrong space). ep0 alone was too tight on z (only reached
−0.04; other episodes lift to 0.088) — caught by the clip-hit-rate test, which is why we scan all
episodes.

**Status:** ✅ end-to-end verified — converter → loader (FK → `action (T,8)`) → **Stage-1 training
runs on the Blackwell GPU** (46.1 M params, 6 steps, 8.5 it/s). Config: `dataset=folding_dataset`
(`action_mode=bimanual_fold`, `action_dim=8`, `obs_keys=[cam_high]`, res 128); `folding_dataset`
registered in `exp_latent_dyn.py`.

**Deferred:** the inverse (`action_primitive_to_joint_pos`) `bimanual_fold` branch + a keyboard
scene mapping are only needed for the interactive keyboard demo / real-robot IK — add at the
inference step, not now.

---

### D2 — Frame rate / `skip_frame`: subsample the action too (keep 8-dim)  *(2026-07-01)*
**Problem:** our qcez data is **50 fps**; the paper's ALOHA data was **~10 Hz** (`--frequency 10`,
`skip_frame=1`). At 50 fps consecutive frames barely move, so a dynamics model would learn trivial
"next≈current" transitions. To match the method's ~10 Hz regime we want `skip_frame=5`.

**The catch:** with `skip_frame>1` the loader keeps every 5th *frame* but **concatenates the 5
skipped actions** (`keys_to_keep_intermediate=["action"]`) → effective `action_dim = skip_frame ×
8 = 40`. The dynamics action-embed is `Linear(8,64)`, so Stage-2 crashes:
`mat1 and mat2 shapes cannot be multiplied (40x40 and 8x64)`.

Why concatenate at all? It's the safe default for **velocity/delta** actions — you can't recover
the intermediate motion from the endpoints. But our action is an **absolute EEF pose target**, so
the transition f0→f5 is fully described by "gripper ends at the pose at f5" — the 4 intermediate
poses are redundant. So we can subsample the action like the frames.

**Options considered:**

| | action_dim | dynamics rate | S1 frames | reconvert? |
|---|---|---|---|---|
| A. skip 5, concat (default) | 40 ✗ | 10 fps | full | no |
| B. skip 1, keep 50 fps | 8 ✓ | 50 fps ✗ | full | no |
| C. pre-subsample data → 10 fps | 8 ✓ | 10 fps | 5× fewer ✗ | yes |
| **D (chosen). subsample the action too** | **8 ✓** | **10 fps ✓** | **full ✓** | **no ✓** |

**Chosen: D.** Remove `"action"` from `keys_to_keep_intermediate` (2 spots in
`real_aloha_dataset.py`: the train sampler ~L467 and the val sampler in `get_validation_dataset`
~L551). Then the action falls into the `else: sample[key][::skip_frame]` branch — every-5th, 8-dim,
aligned with the every-5th frames. Gets all four right: clean 8-dim EEF action, meaningful 10 fps
dynamics, Stage-1 keeps all frames (it uses `skip_frame=1`), no reconversion. Valid *because* the
action is a position target (subsampling is lossless here; would NOT be for velocity/delta actions).

Stage 1 unaffected (single-frame autoencoder, `skip_frame=1`). Stage 2/3 use `skip_frame=5`.

---

## 5. Training stages (per task, single GPU)

| Stage | What | Time (H200 ref) | Notes |
|---|---|---|---|
| 1 | Autoencoder (CNN enc + consistency decoder) | ~6 h | reconstruction must be **near-perfect** before stage 2 |
| 2 | Latent dynamics (action-conditioned next-latent) | ~12 h | the expensive one; autoregressive at inference |
| 3 | Decoder finetune (robust to noisy latents) | < stage 1 | order of 2/3 swappable |

Model is lightweight (~176 MB). Runs/trains on the local **RTX PRO 4000 Blackwell** (cu128);
inference needs only a 2080-class GPU; ~15 fps on a 4090; 10-min stable rollouts.

Two noises in Stage 2: (a) diffusion/consistency denoising noise (generative);
(b) context-noise injection `prev_frame_noise_scale=0.1` for autoregressive-drift robustness.
**Neither fixes coverage** — they stabilize near the demonstrated manifold, not extrapolate.

---

## 5b. Hyperparameters — paper/code vs ours

### Model + optimizer → IDENTICAL (inherited from the unchanged configs)
| Hyperparameter | Paper/code = Ours |
|---|---|
| learning rate | 8e-5 |
| weight decay | 1e-4 |
| warmup steps | 10,000 |
| LR schedule | linear |
| Adam betas | [0.9, 0.999] |
| latent_dim / enc_dim / downsample | 512 / 64 / 2 |
| diffusion | timesteps 1000, sampling 50, β=sigmoid, pred_v |
| decode infer steps | 3 |
| prev-frame noise | 0.1 |
| resolution | 128 |
| precision | 32-true |
| gradient clip | 1.0 |

We did not touch any of these → no tuning divergence.

### Per-stage training knobs → match except batch size (GPU-limited)
| Knob | Paper S1 | Paper S2 | Paper S3 | Ours |
|---|---|---|---|---|
| batch_size | 1 | 4 | 16 | **4** (24 GB cap) |
| horizon | 1 | 10 | 1 | same |
| val_horizon | 1 | 200 | 200 | same |
| max_steps | 1,000,005 | 1,000,005 | 1,000,005 | ceiling + monitor/early-stop |
| val_every | 6,000 | 30,000 | 30,000 | ~3,000 (smaller data) |
| action_dim | 4 (push) | — | — | **8** (= their rope) |

- **Stage 1:** paper used batch **1**; our batch **4 is larger** — fine (more stable; ~19 GB, fits).
- **Stage 2:** paper batch 4 = ours. ⚠️ horizon 10 raises memory — verify batch-4×horizon-10 fits (runs in latent space, lighter than S1 pixel decode).
- **Stage 3:** paper batch **16**; we cap at ~4 → use **gradient accumulation (accum=4 → eff 16)** or more steps. Only forced deviation.

### GPU memory on the RTX PRO 4000 Blackwell (24 GB) — measured
| batch (Stage 1) | memory | fits? |
|---|---|---|
| 4 | ~19 GB | ✅ proven |
| 6–8 | ~20–21 GB | ⚠️ tight (validation FVD+render spikes) |
| 16 | >23.4 GB | ❌ **OOM (measured)** |
Fixed overhead is large (~17–18 GB: model + consistency/diffusion forward + FVD/LPIPS metric models + CUDA ctx), so memory doesn't scale from a low base. **Operating point = batch 4.** Validation is the memory spike (FVD model + `render_img_cm` denoise).

### Data → the substantive differences (affect quality, not the recipe)
| | Paper | Ours |
|---|---|---|
| episodes/task | ~600 | 45 (~13× less) |
| data type | play (wide coverage) | demonstration (narrow) |
| capture rate | ~10 Hz (`--frequency 10`) | 50 fps → **`skip_frame=5`** for S2 (≈10 Hz) |
| cameras | 1 | 1 (cam_high) |
| action_dim | 4 / 8 | 8 (bimanual_fold) |

`skip_frame=5` makes our S2 temporal span = horizon 10 × 5 = 50 native frames ≈ **1 s** — same 1 s context the paper got from horizon 10 @ 10 Hz.

### No automatic early stopping
No `EarlyStopping` callback exists; paper trains to `max_steps` (=∞) and stops manually on validation reconstruction. `ModelCheckpoint` saves every 10,000 steps (no best-tracking). Plan: add `EarlyStopping(monitor="validation/fvd", mode=min, patience)` + `ModelCheckpoint(monitor="validation/fvd", save_top_k=1, save_last=True)` for hands-off, overfit-guarded runs.

### First-pass Stage-1 result (one episode, 800 steps)
Reconstruction MSE 0.090 → 0.075 (step 400 → 800); structure (garment position/size) visibly tracks GT — encoder/decoder confirmed learning. Renderer: `scripts/render_recon.py <run_dir>`. Rough because ~800 steps ≪ full Stage 1.

---

## 5c. Stage 1 full run — status & results

**Run:** all 45 episodes, `dataset=folding_dataset` (`action_mode=bimanual_fold`, `action_dim=8`,
`obs_keys=[cam_high]`, res 128), **batch 4**, val + checkpoint every 3,000 steps (`+save_last=true`),
`max_steps=45,000` ceiling, wandb offline. Launched **detached** (`setsid nohup … &`) so it survives
session teardown. Run dir: `outputs/2026-06-30/23-28-40`. Logs: `outputs/stage1_full/train.log`.

**Convergence (healthy — training fits, validation still improving, no overfitting):**

| metric | step 3k | step 9k | trend |
|---|---|---|---|
| training/rec_loss | (0.066 at start) | **~0.0007** | plateaued ~step 3–4k (90× drop) |
| validation/mse ↓ | 0.052 | **0.033** | dropping, steepening after 6k |
| validation/psnr ↑ | 18.9 dB | **20.9 dB** | rising, steepening |
| validation/ssim ↑ | 0.35 | **0.54** | rising steadily |
| validation/uiqi ↑ | 0.028 | **0.088** | rising |

Train loss plateaued while **val metrics keep climbing (and accelerating)** → generalizing, *not*
overfitting. 45 demo episodes are enough for the autoencoder to generalize to held-out folds.
Near-perfect ≈ PSNR > 30 / SSIM > 0.9; at step 9k we're 21 / 0.54 and rising → keep training.

**Reconstruction visual progression** (`scripts/render_recon.py <run_dir>` → `recon_compare.png`):
800 steps = faint blobs in noise → 9k steps = legible scene (garment shape/position, arms, fold
progression track GT), residual grain. MSE on a 6-frame probe: 0.075 (800) → 0.037 (9k).

**Monitoring tooling (added):**
- `scripts/render_recon.py <run_dir>` — GT-vs-reconstruction grid from latest checkpoint.
- `scripts/plot_loss.py <run_dir>` — extracts scalar history from the **offline** wandb binary
  (`run-*.wandb` via `DataStore`; keys live in `history.item[].nested_key`) → `loss_curves.png`.

**Auto-stop note:** the real Stage-1 monitor metric is **`validation/mse`** (mode=min) or
`validation/psnr` (max) — **not** `validation/fvd` (FVD isn't logged for single-frame Stage 1).
Stop when these plateau; as of step ~12k they have not.

**Gotchas hit & fixed:** (1) long runs die on session teardown → launch detached with `setsid`.
(2) `checkpointing.save_last` needs `+` prefix (not in struct). (3) `use_cache=true` errors on a
stale `camera_1_color` key → use `use_cache=false`. (4) `torch.load` needs `weights_only=False`
(ckpt holds OmegaConf objects). (5) batch 16 OOMs (24 GB) → batch 4.

---

## 6. Data adequacy — honest assessment

iws "typically enough" per task: **~600 play episodes × 200 steps (~6 h)**.

| | This dataset (200_1_14) |
|---|---|
| Quantity | 200 eps / 245k frames — decent; subsampled ~49k effective steps |
| Type | ❌ **demonstration data** (clean successful folds), not **play** data |
| Task | cloth folding = hardest (deformable, self-occlusion) |

**"Wrong type" = coverage, not count.** A world model must be accurate over states/actions a
policy might visit (incl. failures/off-task); demos only show the thin successful manifold. More
demos densify the same tube, don't widen it. → expect plausible folding-like rollouts that snap
back to demonstrated motion; weak on genuinely off-distribution actions.

**Adequate for:** pipeline test, controllability/keyboard demo, matched-policy eval-loop demo.
**Not adequate for:** trustworthy long-horizon closed-loop evaluation.

For a real coverage test, use **self-generated MuJoCo play data**
(`scripts/data_collection/sim_aloha_dataset_collection_scripted.py`, motion types:
linear / rotating / random_contact / random_no_contact) — right type, free, unlimited,
plus ground-truth physics to validate against.

---

## 7. Step-by-step

1. **Download** `qcez/folding_clothes_200_1_14_lerobot` → `data/folding/`.
   (Later: `qcez/fold-cloth-1_14-200-checkpoint` for the eval-loop demo.)
2. **Write converter** (§4.A) + verify: round-trip a couple episodes, render one to MP4
   (cf. `scripts` we used for `bimanual_rope`).
3. **Add loader branch** (§4.B) + dataset config yaml.
4. **Stage 1** training → confirm near-perfect reconstruction on the Blackwell GPU.
5. **Stage 2** dynamics → **Stage 3** decoder finetune.
6. **Inference**: replay-driven rollout first (cleaner than 14-DOF keyboard); add a reduced
   keyboard mapping later if desired.
7. **(Stretch)** drop `fold-cloth-1_14-200-checkpoint` into the trained WM → compare in-sim
   vs. real folding → the full evaluation story.

---

## 8. Open decisions
- [ ] Confirm Option A vs B (default: **A**).
- [ ] Camera: `cam_high` only (default) vs add wrist views.
- [ ] Resolution: 128 (cheap) vs 224 (native).
- [ ] Scope: controllability demo only, or pursue the eval-loop / correlation result.

---

## Reference: useful repo paths
- Dataset loader: `interactive_world_sim/datasets/latent_dynamics/real_aloha_dataset.py`
- Action encoding: `interactive_world_sim/utils/action_utils.py`
- Keyboard teleop: `scripts/inference/teleoperate_keyboard.py`
- Algo config: `configurations/algorithm/latent_world_model.yaml`
- Dataset config: `configurations/dataset/real_aloha_dataset.yaml`
- Data collection (sim play data): `scripts/data_collection/sim_aloha_dataset_collection_scripted.py`
- Repo HF data: `yixuan1999/interactive-world-sim-{min-data,data,mujoco-data}`
