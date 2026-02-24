import torch
import numpy as np
import os
import av
from transformers import AutoImageProcessor, AutoModel
from typing import List, Dict, Union, Tuple
import time
import warnings
import pandas as pd
import pickle
from tqdm import tqdm
warnings.filterwarnings("ignore")

def time_ms():
    return time.time()*1000

with open("../../wandb_username.txt", "r") as f:
    wandb_username = f.read().strip()

with open("../../project_name.txt", "r") as f:
    project_name = f.read().strip()

with open("../../project_dir.txt", "r") as f:
    project_dir = f.read().strip()

sample_video_path = f"/localdisk1/{project_dir}/{project_name}/code/VFMs/sample_data/sample_youtube_video.mp4"

np.random.seed(0)

model_configs = {
        "VideoMAE": {
            "model_name": "MCG-NJU/videomae-base",
            "num_frames": 16,
            "image_size": 224
        }
    }

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

def read_video_pyav(container, indices, reduced_height):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
        reduced_height (`int`): Each frame height is reduced to this height, and width is adjusted maintaining the original aspect ratio
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            old_height, old_width = frame.height, frame.width
            new_height = reduced_height
            downsample_ratio = new_height/old_height
            new_width = int(downsample_ratio*old_width)
            frame = frame.reformat(width=new_width, height=new_height)
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(n_frames_to_sample, stride, n_total_frames):
    '''
    Sample a given number of frame indices from the video.
    Args:
        n_frames_to_sample (`int`): Total number of frames to sample.
        stride (`int`): Sample every n-th frame.
        n_total_frames (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(n_frames_to_sample * stride)
    end_idx = np.random.randint(converted_len, n_total_frames)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=n_frames_to_sample)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def extract_embeddings(model_tag="VideoMAE", file_path=sample_video_path):
    embeddings = {}
    container = av.open(file_path)
                
    # sample n frames
    indices = sample_frame_indices(n_frames_to_sample=model_configs[model_tag]["num_frames"], stride=1, n_total_frames=container.streams.video[0].frames)
    
    # pre-process video size (height and width), convert to numpy
    video = read_video_pyav(container, indices, reduced_height=model_configs[model_tag]["image_size"])
    # print(video.shape) # 16 x 224 x 298 x 3
    
    # Load image processor and pre-trained video encoder model
    image_processor = AutoImageProcessor.from_pretrained(model_configs[model_tag]["model_name"])
    model = AutoModel.from_pretrained(model_configs[model_tag]["model_name"]).to(device)

    # prepare video for the model
    with torch.no_grad():
        inputs = image_processor(list(video), return_tensors="pt").to(device)

        # t1 = time_ms()
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        # print(last_hidden_states.shape) # 1 x 1568 x 768
        # t2 = time_ms()
        # print(f"Runtime: {t2-t1} ms")

    mean_pooled = torch.mean(last_hidden_states, dim=-2).reshape(-1)    # (768,)
    max_pooled = torch.max(last_hidden_states, dim=-2)[0].reshape(-1)   # (768,)
    embeddings["mean_pooled_embedding"] = mean_pooled.cpu()
    embeddings["max_pooled_embedding"] = max_pooled.cpu()
    return embeddings

if __name__ == "__main__":
    # These parameters will be eventually obtained through command line arguments
    model_tag = "VideoMAE"
    file_path = sample_video_path
    embeddings = extract_embeddings(model_tag=model_tag, file_path=file_path)
    print(embeddings)