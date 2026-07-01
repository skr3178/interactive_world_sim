from huggingface_hub import snapshot_download
pats = [f"data/chunk-000/episode_{i:06d}.parquet" for i in range(50,200)]
p = snapshot_download(
    repo_id="qcez/folding_clothes_200_1_14_lerobot",
    repo_type="dataset",
    local_dir="/media/skr/storage/YC/interactive_world_sim/data/folding_raw",
    allow_patterns=pats,
    max_workers=8,
)
print("DONE", p)
