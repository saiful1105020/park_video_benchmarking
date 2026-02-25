#!/usr/bin/env python3
import os
import pickle
import warnings
from typing import List

import av
import click
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))         
VFMS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))     
sys.path.insert(0, VFMS_ROOT)
from model_setup import model_configs

warnings.filterwarnings("ignore")
np.random.seed(0)

MODEL_TAG = "VJEPA2"

HF_TOKEN = os.environ.get("HF_TOKEN", None)

HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", os.path.join(VFMS_ROOT, ".cache", "hf"))
os.makedirs(HF_CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_MODULES_CACHE"] = os.path.join(HF_CACHE_DIR, "modules")

device = "cuda" if torch.cuda.is_available() else "cpu"


def read_video_pyav(container: av.container.InputContainer, indices: np.ndarray, reduced_height: int) -> np.ndarray:
    """
    Decode selected frames via PyAV and resize by height.
    Returns np.ndarray of shape (T, H, W, 3), dtype=uint8 (rgb24).
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

    if len(frames) == 0:
        return np.zeros((0, int(reduced_height), int(reduced_height), 3), dtype=np.uint8)

    return np.stack([x.to_ndarray(format="rgb24") for x in frames], axis=0)


def sample_frame_indices(
    n_frames_to_sample: int,
    stride: int,
    n_total_frames: int,
    n_views: int = 1
) -> List[np.ndarray]:
    """
    Returns a list (length n_views) of index arrays.
    """
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


def load_processor(model_name: str) -> AutoVideoProcessor:
    kwargs = dict(cache_dir=HF_CACHE_DIR)
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    return AutoVideoProcessor.from_pretrained(model_name, **kwargs)


def load_vjepa2_model(model_name: str) -> torch.nn.Module:
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = dict(
        cache_dir=HF_CACHE_DIR,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    model = AutoModel.from_pretrained(model_name, **kwargs)
    return model.to(device).eval()


@click.command()
@click.option("--video_root", default=None,
              help="Directory containing .mp4 files (non-recursive). Default: VFMs/sample_data")
@click.option("--out_root", default=None,
              help="Output directory root. Default: same as --video_root")
@click.option("--save_every", default=100, type=int,
              help="Save progress every N videos.")
@click.option("--stride", default=2, type=int,
              help="Stride for temporal sampling (default: 2).")
@click.option("--num_views", default=4, type=int,
              help="Number of views to extract per video (default: 4).")
def main(video_root: str, out_root: str, save_every: int, stride: int, num_views: int):
    # Portable defaults
    if video_root is None:
        video_root = os.path.join(VFMS_ROOT, "sample_data")
    if out_root is None:
        out_root = video_root

    if MODEL_TAG not in model_configs:
        raise ValueError(f"{MODEL_TAG} not found in model_configs. Available: {list(model_configs.keys())}")

    mc = model_configs[MODEL_TAG]
    model_name = mc["model_name"]

    model_dir = os.path.join(out_root, MODEL_TAG)
    os.makedirs(model_dir, exist_ok=True)

    count_filename = os.path.join(
        model_dir,
        f"{MODEL_TAG}_{num_views}views_{stride}stride_Features_All_PARK_Videos_Count_Completed.txt"
    )
    pkl_filename = os.path.join(
        model_dir,
        f"{MODEL_TAG}_{num_views}views_{stride}stride_Features_All_PARK_Videos.pkl"
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

    processor = load_processor(model_name)
    model = load_vjepa2_model(model_name)

    filenames = [fn for fn in os.listdir(video_root) if fn.lower().endswith(".mp4")]
    print(f"Found {len(filenames)} video files")

    count = 0
    for filename in tqdm(filenames, desc=f"Extracting with {MODEL_TAG}"):
        count += 1
        if count <= completed_count:
            continue

        file_path = os.path.join(video_root, filename)
        container = None

        try:
            container = av.open(file_path)
            stream = container.streams.video[0]
            total_frames = int(stream.frames) if stream.frames else 1000
            if total_frames == 0:
                continue

            multi_view_indices = sample_frame_indices(
                n_frames_to_sample=mc["num_frames"],
                stride=stride,
                n_total_frames=total_frames,
                n_views=num_views
            )

            item = {"filename": filename}

            for view_idx, indices in enumerate(multi_view_indices):
                video_np = read_video_pyav(container, indices, reduced_height=mc["image_size"])  # (T,H,W,3)

                T_needed = mc["num_frames"] # pad (repeat last frame) if fewer than requested frames were decoded
                if video_np.shape[0] == 0:
                    video_np = np.zeros((T_needed, mc["image_size"], mc["image_size"], 3), dtype=np.uint8)
                elif video_np.shape[0] < T_needed:
                    pad_frame = video_np[-1:]
                    pad = np.repeat(pad_frame, T_needed - video_np.shape[0], axis=0)
                    video_np = np.concatenate([video_np, pad], axis=0)

                with torch.no_grad():
                    inputs = processor(list(video_np), return_tensors="pt").to(device)
                    outputs = model(**inputs)

                    if not hasattr(outputs, "last_hidden_state"):
                        raise RuntimeError("Model output does not contain last_hidden_state; cannot pool tokens.")

                    last_hidden_states = outputs.last_hidden_state  # [B, tokens, dim]

                    mean_pooled = last_hidden_states.mean(dim=-2).reshape(-1)  # (D,)
                    max_pooled = last_hidden_states.amax(dim=-2).reshape(-1)   # (D,)

                item[f"view_{view_idx}_mean_pooled_embedding"] = mean_pooled.cpu()
                item[f"view_{view_idx}_max_pooled_embedding"] = max_pooled.cpu()

            dataset.append(item)

        except Exception as e:
            print(f"[WARN] Failed on {filename}: {e}")

        finally:
            try:
                if container is not None:
                    container.close()
            except Exception:
                pass

        # periodic save
        if count % save_every == 0:
            with open(pkl_filename, "wb") as f:
                pickle.dump(dataset, f)
            with open(count_filename, "w") as f:
                f.write(f"{count}\n")
            print(f"Partial progress saved: {count}/{len(filenames)}")

    # final save
    with open(pkl_filename, "wb") as f:
        pickle.dump(dataset, f)
    with open(count_filename, "w") as f:
        f.write(f"{count}\n")

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()