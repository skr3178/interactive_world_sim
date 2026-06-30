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

So `bimanual_fold` uses a box = **our measured range + margin**: `x[0.28,0.66] y[−0.45,0.48]
z[−0.48,0.02]`. Strategy: bounds encompass real data so training actions are **never clipped**
(no distortion); the box only guards out-of-range *control* input. **Caveat:** derived from
episode 0 only — widen/refine after the full 200-episode download.

**Status:** ✅ smoke-tested — loader builds the replay buffer, FK runs, yields `action (T,8)`
with per-dim ranges matching raw FK (clip confirmed no-op on real data).

**Deferred:** the inverse (`action_primitive_to_joint_pos`) `bimanual_fold` branch + a keyboard
scene mapping are only needed for the interactive keyboard demo / real-robot IK — add at the
inference step, not now.

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
