# VMFL: Think-See-Callback

Code and JSONL annotations for *Think-See-Callback: Visual Measurement Feedback for Long-Horizon Spatial Reasoning*.

This repository provides supervised fine-tuning (SFT) and closed-loop evaluation for Maze Planning, Maze Navigation, Mini-BEHAVIOR, and Visual Spatial Planning (VSP). Model weights and images are distributed separately. The paper's reinforcement learning stage and VIMA-Bench experiments are not included.

## Data

| Task | Training records | Test records | JSONL directory |
|---|---:|---:|---|
| Maze Planning | 2,027 | 800 | `data/maze_plan/` |
| Maze Navigation | 5,007 | 1,255 | `data/maze_navigation/` |
| Mini-BEHAVIOR | 6,400 | 1,604 | `data/mini_behavior/` |
| VSP | 1,000 | 400 | `data/vsp/` |

Each directory contains `train.jsonl` and `test.jsonl`. Image paths in the JSONL are relative to the repository root and begin with `data/images/`.

The companion Hugging Face dataset contains image archives for Maze Planning, Maze Navigation, and Mini-BEHAVIOR. Replace `NAMESPACE` with the dataset owner's namespace. For a private dataset, obtain access and run `hf auth login` before downloading.

```bash
export HF_DATASET_REPO=NAMESPACE/VMFL-images
mkdir -p .cache/vmfl-images data/images
hf download "$HF_DATASET_REPO" --repo-type dataset --local-dir .cache/vmfl-images
(cd .cache/vmfl-images && sha256sum -c SHA256SUMS)
for archive in maze_plan_images.tar maze_navigation_images.tar mini_behavior_images.tar; do
  tar -xf ".cache/vmfl-images/$archive" -C data/images
done
```

VSP images are obtained from the [Mirage VSP archives](https://github.com/UMass-Embodied-AGI/Mirage/tree/main/data/vsp_spatial_planning) and are not redistributed here. Restore them after installing the Python dependencies:

```bash
mkdir -p .cache
curl -L --fail -o .cache/vsp_spatial_planning.tar.gz https://raw.githubusercontent.com/UMass-Embodied-AGI/Mirage/53f26de2682025e146781c5b198ec93bdfe4c4d6/data/vsp_spatial_planning/vsp_spatial_planning.tar.gz
curl -L --fail -o .cache/vsp_spatial_planning_test.tar.gz https://raw.githubusercontent.com/UMass-Embodied-AGI/Mirage/53f26de2682025e146781c5b198ec93bdfe4c4d6/data/vsp_spatial_planning/vsp_spatial_planning_test.tar.gz
python restore_vsp_images.py --train-archive .cache/vsp_spatial_planning.tar.gz --test-archive .cache/vsp_spatial_planning_test.tar.gz
```

The VSP training conversations were reconstructed from trajectory annotations. They are not the archived prompts used for the paper's training run. The Maze Navigation test file contains 1,255 records; the option counts in the paper also sum to 1,255, although the printed total is 1,225.

## Installation

Run these commands from the repository root on Linux with Python 3.10 and a CUDA GPU:

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

Use a PyTorch wheel compatible with the installed CUDA driver. Training uses the bundled ModelScope-SWIFT source in `swift/`.

## Training

The scripts use Qwen2.5-VL-3B for Maze Planning and Qwen2.5-VL-7B for the other tasks. The model IDs can be overridden with `MODEL`.

```bash
TASK=maze_plan GPUS=0,1,2,3 bash train.sh
TASK=maze_navigation GPUS=0,1,2,3 bash train.sh
TASK=mini_behavior GPUS=0,1,2,3 bash train.sh
TASK=vsp GPUS=0,1,2,3 DEEPSPEED=zero3_offload bash train.sh
```

Training uses five epochs, a learning rate of `1e-5`, a frozen vision tower, and a global batch size of 16. `OUTPUT` sets the checkpoint directory. `GPUS` may contain one, two, or four device IDs; gradient accumulation is adjusted to keep the global batch size fixed.

## Evaluation

Use the final checkpoint from training. For example, after training Maze Navigation with the default `OUTPUT`:

```bash
CHECKPOINT=$(find outputs/sft_maze_navigation -path '*/checkpoint-*/config.json' | sort -V | tail -n 1)
CHECKPOINT=${CHECKPOINT%/config.json}
test -n "$CHECKPOINT" && TASK=maze_navigation MODEL="$CHECKPOINT" GPUS=0,1,2,3 OUTPUT=results/maze_navigation bash run_eval.sh
```

`run_eval.sh` splits the test set across the listed GPUs and writes `summary.json` in the output directory. Set `TASK` to any of the four task names and choose a checkpoint trained for that task.

## Licenses and citation

VMFL code is under the MIT License. Bundled ModelScope-SWIFT code is under Apache 2.0; see `swift/LICENSE` and `swift/ORIGIN.md`. VMFL annotations are under CC BY 4.0; see `data/LICENSE`. Third-party VSP source material and image assets are excluded from the annotation license.

If you use this repository, cite *Think-See-Callback: Visual Measurement Feedback for Long-Horizon Spatial Reasoning* (WACV 2027).
