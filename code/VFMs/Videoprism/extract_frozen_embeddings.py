#!/usr/bin/env python3
import os
import sys
import pickle
import warnings
import importlib
from pathlib import Path
from typing import List

import av
import click
import numpy as np
from tqdm import tqdm


try:
    cudnn_lib = Path(importlib.import_module("nvidia.cudnn").__file__).with_name("lib")
    os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
except (ImportError, AttributeError):
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          
VFMS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))     
sys.path.insert(0, VFMS_ROOT)
from model_setup import model_configs

warnings.filterwarnings("ignore")
np.random.seed(0)

MODEL_TAG = "VideoPrism"


def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def pil_resize_video(frames_np: np.ndarray, size: int) -> np.ndarray:
    """Resize [T,H,W,3] -> [T,size,size,3] using PIL bilinear."""
    from PIL import Image

    out = []
    for fr in frames_np:
        im = Image.fromarray(fr)
        im = im.resize((size, size), resample=Image.BILINEAR)
        out.append(np.asarray(im, dtype=frames_np.dtype))
    return np.stack(out, axis=0)


def read_video_pyav(container: av.container.InputContainer, indices: np.ndarray, reduced_height: int) -> np.ndarray:
    """
    Decode selected frames via PyAV and resize by height (keep aspect ratio).
    Returns np.ndarray (T, H, W, 3) uint8. If no frames decoded, returns empty (0, ..., 3).
    """
    frames = []
    container.seek(0)

    start_index = int(indices[0])
    end_index = int(indices[-1])
    idx_set = set(int(x) for x in indices)

    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in idx_set:
            old_h, old_w = frame.height, frame.width
            new_h = int(reduced_height)
            ratio = new_h / float(old_h) if old_h > 0 else 1.0
            new_w = max(1, int(round(ratio * old_w)))
            frame = frame.reformat(width=new_w, height=new_h)
            frames.append(frame)

    if not frames:
        return np.zeros((0, int(reduced_height), int(reduced_height), 3), dtype=np.uint8)

    return np.stack([x.to_ndarray(format="rgb24") for x in frames], axis=0)


def sample_frame_indices(
    n_frames_to_sample: int,
    stride: int,
    n_total_frames: int,
    n_views: int = 1
) -> List[np.ndarray]:
    """Return list (length n_views) of sampled frame-index arrays."""
    if n_total_frames == 0:
        raise Exception("no frame in the video")

    indices = []
    converted_len = int(n_frames_to_sample * stride)
    multi_view_converted_len = converted_len * n_views

    if multi_view_converted_len <= n_total_frames:
        end_idx = np.random.randint(multi_view_converted_len, n_total_frames + 1)
        start_idx = end_idx - multi_view_converted_len

        while start_idx < end_idx:
            local_end_index = start_idx + converted_len
            view_indices = np.linspace(
                start_idx, local_end_index, num=n_frames_to_sample + 1
            )[:-1].astype(np.int64)
            indices.append(view_indices)
            start_idx = local_end_index
    else:
        if n_total_frames < converted_len:
            raise Exception("do not have enough frames for a single view")
        start_idx = 0
        while len(indices) < n_views:
            local_end_index = start_idx + converted_len
            view_indices = np.linspace(
                start_idx, local_end_index, num=n_frames_to_sample + 1
            )[:-1].astype(np.int64)
            indices.append(view_indices)

            if (local_end_index + converted_len) < n_total_frames:
                start_idx = local_end_index
            else:
                start_idx = np.random.randint(start_idx, (n_total_frames - converted_len) + 1)

    return indices


def init_videoprism(mc: dict):
    """Initialize VideoPrism model + JIT forward fn (same logic as your working version)."""
    try:
        import jax
        import jax.numpy as jnp
        from jax.extend import backend
        from videoprism import models as vp
    except Exception as e:
        raise RuntimeError(
            f"VideoPrism dependencies missing: {e}\n"
            "Install:\n"
            "  pip install git+https://github.com/google-deepmind/videoprism\n"
            "  pip install --upgrade \"jax[cuda12_pip]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        )

    print(f"JAX version: {jax.__version__}")
    print(f"JAX platform: {backend.get_backend().platform}")
    print(f"JAX devices: {jax.device_count()}")

    model_name = mc["model_name"]
    flax_model = vp.get_model(model_name, fprop_dtype=None)
    loaded_state = vp.load_pretrained_weights(model_name)

    def forward_fn(inputs, *, train: bool = False):
        return flax_model.apply(loaded_state, inputs, train=train)

    forward_jit = jax.jit(forward_fn, static_argnames=("train",))
    print("[VideoPrism] Model loaded & JIT-compiled.")
    return forward_jit, jnp


@click.command()
@click.option("--video_root", default=None,
              help="Directory containing .mp4 files (non-recursive). Default: VFMs/sample_data")
@click.option("--out_root", default=None,
              help="Output directory root. Default: same as --video_root")
@click.option("--save_every", default=100, type=int,
              help="Save progress every N videos.")
@click.option("--max_videos", default=0, type=int,
              help="Process at most this many NEW videos this run (0 = no limit).")
@click.option("--stride", default=2, type=int,
              help="Stride for temporal sampling (default: 2).")
@click.option("--num_views", default=4, type=int,
              help="Number of views to extract per video (default: 4).")
def main(video_root: str, out_root: str, save_every: int, max_videos: int, stride: int, num_views: int):
    if video_root is None:
        video_root = os.path.join(VFMS_ROOT, "sample_data")
    if out_root is None:
        out_root = video_root

    mc = model_configs.get(MODEL_TAG, {"model_name": "videoprism_public_v1_base", "num_frames": 16, "image_size": 288})

    model_dir = os.path.join(out_root, MODEL_TAG)
    safe_mkdir(model_dir)

    count_filename = os.path.join(
        model_dir, f"{MODEL_TAG}_{num_views}views_{stride}stride_Features_All_PARK_Videos_Count_Completed.txt"
    )
    pkl_filename = os.path.join(
        model_dir, f"{MODEL_TAG}_{num_views}views_{stride}stride_Features_All_PARK_Videos.pkl"
    )

    completed_count = 0
    dataset = []
    if os.path.exists(count_filename) and os.path.exists(pkl_filename):
        print("Resuming from previous progress...")
        with open(count_filename, "r") as f:
            line = f.readline().strip()
            if line:
                completed_count = int(line)
        with open(pkl_filename, "rb") as f:
            dataset = pickle.load(f)
        print(f"Completed count: {completed_count}, loaded items: {len(dataset)}")

    videoprism_forward_fn, jnp = init_videoprism(mc)

    filenames = [fn for fn in os.listdir(video_root) if fn.lower().endswith(".mp4")]
    total_files = len(filenames)
    print(f"Found {total_files} video files in {video_root}")

    count = 0
    processed_this_run = 0

    for filename in tqdm(filenames, desc=f"Extracting with {MODEL_TAG}"):
        count += 1
        if count <= completed_count:
            continue

        if max_videos and processed_this_run >= max_videos:
            print(f"Reached --max_videos={max_videos}. Stopping early.")
            break

        file_path = os.path.join(video_root, filename)
        container = None

        try:
            container = av.open(file_path)
            vf = container.streams.video[0] if container.streams.video else None
            n_total_frames = int(vf.frames) if (vf and vf.frames) else 0
            if n_total_frames == 0:
                continue

            multi_view_indices = sample_frame_indices(
                n_frames_to_sample=mc["num_frames"],
                stride=stride,
                n_total_frames=n_total_frames,
                n_views=num_views
            )

            item = {"filename": filename}

            for view_idx, indices in enumerate(multi_view_indices):
                video_np = read_video_pyav(container, indices, reduced_height=mc["image_size"])  # (T,H,W,3)

                T_needed = mc["num_frames"]
                if video_np.shape[0] == 0:
                    video_np = np.zeros((T_needed, mc["image_size"], mc["image_size"], 3), dtype=np.uint8)
                elif video_np.shape[0] < T_needed:
                    pad = np.repeat(video_np[-1:], T_needed - video_np.shape[0], axis=0)
                    video_np = np.concatenate([video_np, pad], axis=0)

                if (video_np.shape[1] != mc["image_size"]) or (video_np.shape[2] != mc["image_size"]):
                    video_np = pil_resize_video(video_np, mc["image_size"])

                video_f = (video_np.astype(np.float32) / 255.0)[None, ...] # normalize [0,1] and add batch dim: [1, T, H, W, 3]
                video_jax = jnp.asarray(video_f)

                embeddings, _aux = videoprism_forward_fn(video_jax, train=False)
                emb = np.asarray(embeddings)

                if emb.ndim == 2:  # (1, D)
                    mean_pooled = emb[0]
                    max_pooled = emb[0]
                elif emb.ndim == 3:  # (1, Tokens, D)
                    mean_pooled = emb.mean(axis=1).reshape(-1)
                    max_pooled = emb.max(axis=1).reshape(-1)
                else:
                    raise ValueError(f"Unexpected VideoPrism embedding shape: {emb.shape}")

                item[f"view_{view_idx}_mean_pooled_embedding"] = np.asarray(mean_pooled)
                item[f"view_{view_idx}_max_pooled_embedding"] = np.asarray(max_pooled)

            dataset.append(item)
            processed_this_run += 1

        except Exception as e:
            print(f"[WARN] Failed on {filename}: {e}")

        finally:
            try:
                if container is not None:
                    container.close()
            except Exception:
                pass

        if count % save_every == 0:
            with open(pkl_filename, "wb") as f:
                pickle.dump(dataset, f)
            with open(count_filename, "w") as f:
                f.write(f"{count}\n")
            print(f"Partial progress saved: {count}/{total_files}")

    with open(pkl_filename, "wb") as f:
        pickle.dump(dataset, f)
    with open(count_filename, "w") as f:
        f.write(f"{count}\n")

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()