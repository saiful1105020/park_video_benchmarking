# Navigating this Repository

## Code
<ul>
  <li><b>Cleanup</b> contains some bookkeeping code to clean up wandb run details.</li>

  <li><b>R1_Dataset</b> contains codes for generating dataset descriptions, including demographic distributions.</li>

  <li><b>R2_Task_...</b> contains codes for training single and multi view models without oversampling. This also contains code to automatically read wandb results, select the best model, re-run that model with 30 random seeds, and generate 95\% confidence intervals. All the results are automatically saved as Latex tables, which directly presented in the paper (with some formatting adjustment).</li>

  <li><b>R3_Screening_...</b> --> Very similar to the last setup (R2), but this time we are using oversampling as our dataset is slightly imbalanced.</li>

  <li><b>R4_Model_...</b> --> This generate the model comparsion figure we have provided in the paper.</li>

  <li><b>Utils</b> contains some library codes that is used throughout the repository.</li>

  <li><b>VFMs</b> contains the instructions for downloading pre-trained weights and extracting frozen embeddings for the models we have experimented with. Please refer to the README files for each specific model for details.</li>
</ul>

### Note 
The ```code/``` folder expects four text files:
```project_dir.txt```, ```project_name.txt```, ```wandb_username.txt```, and ```protocol_name.txt``` -- each containing a string indicating the parent project directory, project's name, the username in wandb.ai, and the name of an internal study protocol. These files are not included in this GitHub repo to comply with anonymization requirements.

The repository will be updated to include these files upon manuscript acceptance.

You can re-create these text files with ```[dummy_values]``` using the following bash commands inside the ```code/``` folder.

```
echo "[PROJECT_DIR]" > project_dir.txt
echo "[PROJECT_NAME]" > project_name.txt
echo "[WANDB_USERNAME]" > wandb_username.txt
echo "[PROTOCOL_NAME]" > protocol_name.txt
```

## Data
Contains necessary metadata, including PD/Non-PD labels, PD stage, and countries from where participants are recruited.

This folder should also contain the frozen embeddings, which are currently absent due to GitHub size limit (details below).

### Frozen Embeddings for Running Benchmark
Since the frozen embeddings from all the videos are large in size, we have uploaded the data in the Box folder.
These data are not available on GitHub.
Please download the necessary data from this Box folder, and replace the data/ folder.

<b>Download Embeddings:</b> 
<a href="https://rochester.box.com/s/utw6cxrodcixenks1dxfbo9kagal7djp">Box Link</a>

### Additional MediaPipe Landmarks
We have also shared additional landmarks (frame-by-frame) for each video of this dataset (when landmarks were successfully extracted). Please consider the task as context to see whether the landmarks could be useful for your research. For example, in a facial expression task, hand landmarks may be uninformative (though they remain available in this repository).

Note that the size of these data might be quite large (~20 GB), so make sure your downloader can resume if the download is paused for any reason.

<b>Download MediaPipe Landmarks:</b> 
<a href="https://rochester.box.com/s/f0d8n26j67pllxu6pjopmbroqdn16bz9">Box Link</a>

We are actively uploading data. Please check again later in time if you do not find what you are looking for. Alternatively, you can also contact the authors for an update.

## Results
Contains the figures and tables used in the paper.

---

# Video Foundation Model Configurations and Sizes

For reproducibility of our work, we report the Hugging Face (HF) or relevant model identifier (or checkpoint name), input image dimension, number of frames per view, feature embedding dimension, and model size in parameters.

| **Model** | **HF Model Name** | **Image Dimension** | **Frames** | **Feature Dim** | **Size (#params)** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| TimeSformer | [`facebook/timesformer-base-finetuned-k400`](https://huggingface.co/facebook/timesformer-base-finetuned-k400) | 224 | 32 | 768 | 121.4M |
| VideoMAE | [`MCG-NJU/videomae-base`](https://huggingface.co/MCG-NJU/videomae-base) | 224 | 16 | 768 | 94.2M |
| ViViT | [`google/vivit-b-16x2-kinetics400`](https://huggingface.co/google/vivit-b-16x2-kinetics400) | 224 | 32 | 768 | 88.7M |
| VJEPA2_ssv2 | [`facebook/vjepa`](https://huggingface.co/facebook/vjepa) | 384 | 32 | 1408 | 1B |
| VJEPA2 | [`facebook/vjepa`](https://huggingface.co/facebook/vjepa) | 256 | 32 | 1408 | 1B |
| VideoMAEv2 | [`OpenGVLab/VideoMAEv2-Large`](https://huggingface.co/OpenGVLab/VideoMAEv2-Large) | 224 | 16 | 1024 | 0.3B |
| VideoPrism | [`google-research/videoprism`](https://github.com/google-research/videoprism) | 288 | 16 | 1024 | 354M |

---

# Hyperparameter Search using Weights and Biases

This project uses Weights & Biases (W&B) to automate the search for optimal hyperparameters. We focus on maximizing dev_auroc for 16 tasks across multiple video architectures.

🚀 sweep_config.yaml
```
program: [python_filename_for_training]
method: random
metric:
  goal: maximize
  name: dev_auroc

parameters:
  # Task & Environment
  task_name:
    value: [task_name]
  seed:
    distribution: int_uniform
    max: 99999
    min: 1
  enable_wandb:
    value: true
  detailed_logs:
    value: false

  # Architecture
  model:
    values:
      - ViViT
      - VideoMAE
      - TimeSformer
      - VideoPrism
      - VideoMAEv2
      - VJEPA2
      - VJEPA2_SSV2
  hidden_dim:
    values: [256, 512, 768, 1024]
  pooling:
    values: [max, mean]
  drop_prob:
    distribution: uniform
    max: 0.5
    min: 0.1

  # Training Hyperparameters
  batch_size:
    value: 64
  learning_rate:
    distribution: log_uniform_values
    max: 0.01
    min: 1e-05
  num_epochs:
    distribution: int_uniform
    max: 200
    min: 20
  optimizer:
    values: [AdamW, SGD]

  # Scheduler
  use_scheduler:
    values: ["yes", "no"]
  scheduler:
    values: [step, reduce]

  # Data Config
  num_views:
    value: [1 (single-view) | 4 (multi-view)]
  view_index:
    value: [0 (single-view) | -1 (multi-view)]
```
🚀 Quick Start: Running the Sweep

Initialize the Sweep (Bash):
```
wandb sweep sweep_config.yaml
```

Start an Agent (Bash):
```
wandb agent <sweep_id>
```
---
