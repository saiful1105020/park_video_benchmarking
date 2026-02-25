# Frozen Embedding Extraction for VJEPA2

The `requirements.txt` file lists the Python environment necessary to extract frozen embeddings for the VJEPA2 model.

The script `extract_frozen_embeddings.py` contains a working demo for multi-view frozen embedding extraction (mean-pooled and max-pooled embeddings per view).

A sample video is provided (downloaded from YouTube): `../sample_data/sample_youtube_video.mp4`.  
Note that this does **not** represent any video from our dataset (we are unable to share raw videos to comply with patient privacy).

## Installation

```bash
conda create -n vjepa2 python=3.10 -y
conda activate vjepa2
pip install -r requirements.txt
```
## Usage (sample video)

From the code/VFMs/VJEPA2/ directory:

```bash
CUDA_VISIBLE_DEVICES=0 \
python extract_frozen_embeddings.py \
  --video_root ../sample_data \
  --out_root ../sample_data \
  --num_views 4 \
  --stride 2 \
  --save_every 1 \
|& tee ../sample_data/run_vjepa2.log
```

## Using other VJEPA2 checkpoints

This script uses the VJEPA2 entry in code/VFMs/model_setup.py (it reads model_configs["VJEPA2"]).
To run a different VJEPA2 checkpoint, edit that config to point to the model you want.

Examples (choose one by commenting/uncommenting):
```bash
# In model_setup.py

# Option A: 384px, 64 frames (SSV2)
"VJEPA2": {
    "model_name": "facebook/vjepa2-vitg-fpc64-384-ssv2",
    "num_frames": 64,
    "image_size": 384,
},

# Option B: 256px, 32 frames
"VJEPA2": {
    "model_name": "facebook/vjepa2-vitg-fpc64-256",
    "num_frames": 32,
    "image_size": 256,
},
```
After changing model_setup.py, rerun the same command. 

**Note:** `model_setup.py` can only have **one** `"VJEPA2"` entry at a time. Keep the checkpoint you want and comment out (or delete) the other, otherwise the later one will overwrite the earlier one.

