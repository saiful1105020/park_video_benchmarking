# Frozen Embedding Extraction for VideoMAEv2

The `requirements.txt` file lists the Python environment necessary to extract frozen embeddings for the VideoMAEv2 model.

The script `extract_frozen_embeddings.py` contains a working demo for multi-view frozen embedding extraction (mean-pooled and max-pooled embeddings per view).

A sample video is provided (downloaded from YouTube): `../sample_data/sample_youtube_video.mp4`.  
Note that this does **not** represent any video from our dataset (we are unable to share raw videos to comply with patient privacy).

## Installation

```bash
conda create -n videomaev2 python=3.10 -y
conda activate videomaev2
pip install -r requirements.txt
```
Note: transformers==4.40.0 is required. Newer versions may cause compatibility issues with VideoMAEv2.
## Usage

```bash
python extract_frozen_embeddings.py \
  --video_root ../sample_data/ \
  --out_root ../sample_data/ \
  --num_views 4 \
  --stride 2 \
  --save_every 1
```

