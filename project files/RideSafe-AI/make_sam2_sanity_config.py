from pathlib import Path
import yaml


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\AI Projects\HelmetViolationSystem")

CONFIG = (
    PROJECT_DIR
    / "sam2_repo"
    / "sam2"
    / "configs"
    / "sam2.1_training"
    / "sam2.1_hiera_s_sanity.yaml"
)

SANITY_DIR = PROJECT_DIR / "sam2_training_sanity"


# ============================================================
# LOAD CONFIG
# ============================================================

if not CONFIG.exists():
    raise FileNotFoundError(
        f"Config not found:\n{CONFIG}"
    )

data = yaml.safe_load(
    CONFIG.read_text(encoding="utf-8")
)


# ============================================================
# SANITY TRAINING SETTINGS
# ============================================================

data["scratch"]["resolution"] = 1024
data["scratch"]["train_batch_size"] = 1
data["scratch"]["num_train_workers"] = 2

# Each sequence contains 8 frames.
# Use only 2 frames per training sample for sanity testing.
data["scratch"]["num_frames"] = 2
data["scratch"]["max_num_objects"] = 3


# ============================================================
# DATASET
# ============================================================

data["dataset"]["img_folder"] = str(
    SANITY_DIR / "JPEGImages"
).replace("\\", "/")

data["dataset"]["gt_folder"] = str(
    SANITY_DIR / "Annotations"
).replace("\\", "/")

data["dataset"]["file_list_txt"] = str(
    SANITY_DIR / "sanity_train.txt"
).replace("\\", "/")

data["dataset"]["multiplier"] = 1


# ============================================================
# SAM 2.1 SMALL ARCHITECTURE
# ============================================================

model = data["trainer"]["model"]

trunk = model["image_encoder"]["trunk"]

trunk["embed_dim"] = 96
trunk["num_heads"] = 1
trunk["drop_path_rate"] = 0.1

trunk["stages"] = [
    1,
    2,
    11,
    2
]

trunk["global_att_blocks"] = [
    7,
    10,
    13
]

trunk["window_pos_embed_bkg_spatial_size"] = [
    7,
    7
]


# ============================================================
# SMALL FPN CHANNELS
# ============================================================

model["image_encoder"]["neck"]["backbone_channel_list"] = [
    768,
    384,
    192,
    96
]


# ============================================================
# SAM 2.1 SMALL CHECKPOINT
# ============================================================

checkpoint_initializer = data["trainer"]["checkpoint"][
    "model_weight_initializer"
]

# IMPORTANT:
# load_state_dict_into_model expects a state_dict directly.
# We need the checkpoint-loading function because our config
# supplies checkpoint_path.

checkpoint_initializer["_target_"] = (
    "training.utils.checkpoint_utils."
    "load_checkpoint_and_apply_kernels"
)

checkpoint_initializer["checkpoint_path"] = (
    "./checkpoints/sam2.1_hiera_small.pt"
)

# The SAM 2.1 checkpoint stores the model weights under "model".
checkpoint_initializer["ckpt_state_dict_keys"] = (
    "model",
)


# ============================================================
# TRAINING LIMIT
# ============================================================

# One epoch only for sanity testing.
data["trainer"]["max_epochs"] = 1


# ============================================================
# WINDOWS DISTRIBUTED BACKEND
# ============================================================

# Windows PyTorch builds do not provide NCCL.
# Use Gloo for the sanity run.

data["trainer"]["distributed"]["backend"] = "gloo"


# ============================================================
# LOCAL GPU
# ============================================================

data["launcher"]["num_nodes"] = 1
data["launcher"]["gpus_per_node"] = 1


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

data["launcher"]["experiment_log_dir"] = (
    "./sam2_logs/sanity_small"
)


# ============================================================
# SAVE CONFIG
# ============================================================

CONFIG.write_text(
    yaml.safe_dump(
        data,
        sort_keys=False
    ),
    encoding="utf-8"
)


# ============================================================
# DISPLAY
# ============================================================

print("=" * 70)
print("SAM2.1 SMALL SANITY CONFIG CREATED")
print("=" * 70)
print()

print("Config:")
print(CONFIG)
print()

print("Model: SAM 2.1 Hiera Small")
print("Checkpoint: sam2.1_hiera_small.pt")

print(
    "Resolution:",
    data["scratch"]["resolution"]
)

print(
    "Batch size:",
    data["scratch"]["train_batch_size"]
)

print(
    "Frames:",
    data["scratch"]["num_frames"]
)

print(
    "Max objects:",
    data["scratch"]["max_num_objects"]
)

print(
    "Epochs:",
    data["trainer"]["max_epochs"]
)

print(
    "Checkpoint loader:",
    checkpoint_initializer["_target_"]
)

print(
    "Checkpoint state key:",
    checkpoint_initializer["ckpt_state_dict_keys"]
)

print(
    "Distributed backend:",
    data["trainer"]["distributed"]["backend"]
)

print(
    "GPUs:",
    data["launcher"]["gpus_per_node"]
)

print()

print("Dataset:")
print(SANITY_DIR)

print()

print("=" * 70)