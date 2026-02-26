# Frozen Embedding Extraction for VideoPrism

The `requirements.txt` file lists the Python environment necessary to extract frozen embeddings for the VideoPrism model.

The script `extract_frozen_embeddings.py` contains a working demo for multi-view frozen embedding extraction (mean-pooled and max-pooled embeddings per view).

A sample video is provided (downloaded from YouTube): `../sample_data/sample_youtube_video.mp4`.  
Note that this does **not** represent any video from our dataset (we are unable to share raw videos to comply with patient privacy).

## Installation

```bash
conda create -n videoprism python=3.10 -y
conda activate videoprism
pip install -r requirements.txt
```
## Usage (sample video)

From the code/VFMs/VideoPrism/ directory:
```bash
CUDA_VISIBLE_DEVICES=0 \
python extract_frozen_embeddings.py \
  --video_root ../sample_data \
  --out_root ../sample_data \
  --num_views 4 \
  --stride 2 \
  --save_every 1 \
  --max_videos 2 \
|& tee ../sample_data/run_videoprism.log
```