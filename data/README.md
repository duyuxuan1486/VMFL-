# Data layout

Each task directory contains `train.jsonl` and `test.jsonl`. Records contain prompts, annotations, and image paths relative to the repository root. Image files are stored separately under `data/images/` and are ignored by Git.

```text
data/
  maze_plan/{train,test}.jsonl
  maze_navigation/{train,test}.jsonl
  mini_behavior/{train,test}.jsonl
  vsp/{train,test}.jsonl
  images/
    maze_plot_plan_large/...
    maze_plot_simulation/...
    mini_behavior/...
    vsp_train_redline_opencv_reverse/...
    Visual-Spatial-Planning/...
```

The first three image directories are supplied as separate archives. VSP images must be restored from the Mirage archives using `restore_vsp_images.py`. See the root README for download and restoration commands.

VSP training conversations were reconstructed from trajectory annotations. The VSP JSONL files include third-party source material; see `LICENSE` for the scope of the annotation license.
