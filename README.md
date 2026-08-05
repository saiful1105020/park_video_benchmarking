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

## Face Landmarks Data

<b>Download MediaPipe Face Landmarks</b>:
<a href="https://rochester.box.com/s/7ipdcagm8u7boufb3q7meeteoiqkl5uu">Box Link</a>

The landmarks are compressed for storage efficiency. Use the <a href="https://github.com/saiful1105020/park_video_benchmarking/blob/main/code/R6_Data_Sharing/read_compressed_face_landmarks.py"> custom script </a> to read the coordinates.

## Hand Keypoint Data

<b>Download MediaPipe Hands Landmarks</b>[download the .pdf file and rename the extension to .zip]:
<a href="https://rochester.box.com/s/x473me47g99g8m8prsbzlaaaru1b29vq">Box Link</a>

The following data description was generated using ChatGPT. The description is verified by the authors. Please use the provided **Python** sample codes with caution.

#### Overview

This directory contains frame-level hand keypoints extracted from source videos using the MediaPipe Hand Landmarker Tasks API. Each JSON file stores:

- Video metadata
- Hand-landmarker configuration
- One record for every processed video frame
- Zero, one, or two detected hands per frame
- 21 image-space landmarks for each detected hand
- 21 world-space landmarks for each detected hand

Frames are retained even when no hand is detected. In that case, the frame record contains an empty `hands` list.

### Directory Contents

```text
hand_keypoints/
├── README.md
├── *.json
└── ...
```

Each JSON file describes the hand detections for one source video. The original source-video filename is stored in `metadata.video_name`.

### JSON Structure

The top-level structure is:

```json
{
  "metadata": {
    "...": "video and extraction metadata"
  },
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0,
      "hands": []
    }
  ]
}
```

### Metadata

The `metadata` object describes the source video and the extraction settings.

| Field | Type | Description |
|---|---:|---|
| `video_name` | string | Filename of the source video. |
| `fps` | number | Source-video frame rate in frames per second. |
| `width` | integer | Video-frame width in pixels. |
| `height` | integer | Video-frame height in pixels. |
| `duration_sec` | number | Video duration in seconds. |
| `frame_count_reported_by_opencv` | integer | Frame count reported by OpenCV for the source video. |
| `frame_count_processed` | integer | Number of frames read and processed by the extraction script. |
| `max_num_hands` | integer | Maximum number of hands that the detector was configured to return per frame. |
| `mediapipe_model` | string | MediaPipe model/API used for extraction. |
| `running_mode` | string | MediaPipe inference mode. The sample uses `VIDEO`. |
| `min_hand_detection_confidence` | number | Minimum confidence required for initial hand detection. |
| `min_hand_presence_confidence` | number | Minimum confidence required for hand presence. |
| `min_tracking_confidence` | number | Minimum confidence required for tracking between frames. |
| `landmark_format` | object | Human-readable descriptions of the coordinate formats stored in each hand record. |

Extraction settings are stored in every file. Code that processes multiple files should read these values from each file rather than assuming that all files use identical settings.

### Frame Records

Each element of `frames` has the following structure:

```json
{
  "frame_index": 7,
  "timestamp_ms": 467,
  "hands": [
    {
      "hand_index": 0,
      "handedness": "Right",
      "handedness_score": 0.8489,
      "image_landmarks": [],
      "world_landmarks": []
    }
  ]
}
```

| Field | Type | Description |
|---|---:|---|
| `frame_index` | integer | Zero-based index of the processed frame. |
| `timestamp_ms` | integer | Frame timestamp in milliseconds. |
| `hands` | array | Hand detections for the frame. The array is empty when no hand is detected. |

For a constant-frame-rate video, the timestamp is approximately:

```text
timestamp_ms ≈ round(1000 × frame_index / fps)
```

Small differences can occur because of rounding or source-video timing.

### Hand Records

Each item in a frame's `hands` array represents one detected hand.

| Field | Type | Description |
|---|---:|---|
| `hand_index` | integer | Position of the hand detection within the current frame's `hands` array. |
| `handedness` | string | Predicted handedness label, typically `Left` or `Right`. |
| `handedness_score` | number | Confidence score for the predicted handedness. |
| `image_landmarks` | array | Twenty-one landmarks in normalized image coordinates, with pixel coordinates included. |
| `world_landmarks` | array | Twenty-one landmarks in estimated three-dimensional world coordinates. |

#### Important tracking note

`hand_index` is local to one frame. It is not a persistent identity or tracking ID. For example, `hand_index: 0` in two consecutive frames is not guaranteed to represent the same physical hand.

The handedness label is also a model prediction and can change between adjacent frames, especially when confidence is low, the hand is partially visible, or the image is ambiguous. Downstream analyses that require stable hand identity should add temporal association or smoothing.

### Image Landmarks

Each detected hand has 21 `image_landmarks`.

```json
{
  "landmark_index": 0,
  "x": 0.3951,
  "y": 0.6441,
  "z": -0.00000002,
  "x_px": 252.83,
  "y_px": 309.18
}
```

| Field | Type | Description |
|---|---:|---|
| `landmark_index` | integer | Anatomical landmark index from 0 through 20. |
| `x` | number | Horizontal coordinate normalized by image width. |
| `y` | number | Vertical coordinate normalized by image height. |
| `z` | number | Relative depth estimated by MediaPipe. Smaller values indicate points closer to the camera. |
| `x_px` | number | Horizontal coordinate in pixels. |
| `y_px` | number | Vertical coordinate in pixels. |

The pixel coordinates are computed as:

```text
x_px = x × width
y_px = y × height
```

Pixel coordinates are stored as floating-point values and are not rounded to integer pixel locations.

In image coordinates:

- The origin is at the upper-left corner.
- `x` increases from left to right.
- `y` increases from top to bottom.
- Values near `(0, 0)` are near the upper-left corner.
- Values near `(1, 1)` are near the lower-right corner.

Normalized coordinates may occasionally fall slightly outside `[0, 1]` when a predicted landmark lies beyond the visible frame boundary.

### World Landmarks

Each detected hand also has 21 `world_landmarks`.

```json
{
  "landmark_index": 0,
  "x": -0.0699,
  "y": 0.0417,
  "z": 0.0476
}
```

| Field | Type | Description |
|---|---:|---|
| `landmark_index` | integer | Anatomical landmark index from 0 through 20. |
| `x` | number | Estimated three-dimensional x coordinate in meters. |
| `y` | number | Estimated three-dimensional y coordinate in meters. |
| `z` | number | Estimated three-dimensional z coordinate in meters. |

World landmarks are model estimates. They are not a substitute for calibrated motion-capture measurements, and their accuracy can be affected by occlusion, image quality, hand orientation, camera viewpoint, and distance from the camera.

### Landmark Index Map

MediaPipe uses the following 21-point hand topology:

| Index | Landmark |
|---:|---|
| 0 | Wrist |
| 1 | Thumb CMC |
| 2 | Thumb MCP |
| 3 | Thumb IP |
| 4 | Thumb tip |
| 5 | Index-finger MCP |
| 6 | Index-finger PIP |
| 7 | Index-finger DIP |
| 8 | Index-finger tip |
| 9 | Middle-finger MCP |
| 10 | Middle-finger PIP |
| 11 | Middle-finger DIP |
| 12 | Middle-finger tip |
| 13 | Ring-finger MCP |
| 14 | Ring-finger PIP |
| 15 | Ring-finger DIP |
| 16 | Ring-finger tip |
| 17 | Little-finger MCP |
| 18 | Little-finger PIP |
| 19 | Little-finger DIP |
| 20 | Little-finger tip |

Abbreviations:

- `CMC`: carpometacarpal joint
- `MCP`: metacarpophalangeal joint
- `IP`: interphalangeal joint
- `PIP`: proximal interphalangeal joint
- `DIP`: distal interphalangeal joint

### Missing Detections

A frame with no detected hand is represented as:

```json
{
  "frame_index": 0,
  "timestamp_ms": 0,
  "hands": []
}
```

An empty `hands` list does not necessarily mean that no hand was present in the source frame. It means that no hand satisfied the detector's confidence and tracking criteria for that frame.

Do not remove these frames automatically. They preserve the original timeline and may be useful for:

- Measuring detection coverage
- Identifying periods of occlusion or poor visibility
- Aligning keypoints with the source video
- Distinguishing missing detections from missing frame records

### Loading a File in Python

```python
import json
from pathlib import Path

json_path = Path("hand_keypoints/example.json")

with json_path.open("r", encoding="utf-8") as file:
    data = json.load(file)

metadata = data["metadata"]
frames = data["frames"]

print(f"Video: {metadata['video_name']}")
print(f"Resolution: {metadata['width']} x {metadata['height']}")
print(f"FPS: {metadata['fps']}")
print(f"Processed frames: {len(frames)}")

detected_frame_count = sum(bool(frame["hands"]) for frame in frames)
coverage = detected_frame_count / len(frames) if frames else 0.0

print(f"Frames with at least one detected hand: {detected_frame_count}")
print(f"Frame-level detection coverage: {coverage:.2%}")
```

### Iterating Through Landmarks

```python
for frame in data["frames"]:
    frame_index = frame["frame_index"]
    timestamp_ms = frame["timestamp_ms"]

    for hand in frame["hands"]:
        handedness = hand["handedness"]
        handedness_score = hand["handedness_score"]

        for landmark in hand["image_landmarks"]:
            landmark_index = landmark["landmark_index"]
            x = landmark["x"]
            y = landmark["y"]
            z = landmark["z"]
            x_px = landmark["x_px"]
            y_px = landmark["y_px"]

            # Add task-specific processing here.
```

### Converting Image Landmarks to a Table

The following example creates one row per frame, hand, and landmark:

```python
import json
import pandas as pd
from pathlib import Path

json_path = Path("hand_keypoints/example.json")

with json_path.open("r", encoding="utf-8") as file:
    data = json.load(file)

rows = []

for frame in data["frames"]:
    for hand in frame["hands"]:
        world_by_index = {
            landmark["landmark_index"]: landmark
            for landmark in hand["world_landmarks"]
        }

        for image_landmark in hand["image_landmarks"]:
            index = image_landmark["landmark_index"]
            world_landmark = world_by_index[index]

            rows.append(
                {
                    "video_name": data["metadata"]["video_name"],
                    "frame_index": frame["frame_index"],
                    "timestamp_ms": frame["timestamp_ms"],
                    "hand_index": hand["hand_index"],
                    "handedness": hand["handedness"],
                    "handedness_score": hand["handedness_score"],
                    "landmark_index": index,
                    "x": image_landmark["x"],
                    "y": image_landmark["y"],
                    "z": image_landmark["z"],
                    "x_px": image_landmark["x_px"],
                    "y_px": image_landmark["y_px"],
                    "world_x_m": world_landmark["x"],
                    "world_y_m": world_landmark["y"],
                    "world_z_m": world_landmark["z"],
                }
            )

landmarks_df = pd.DataFrame(rows)
print(landmarks_df.head())
```

Frames without a detected hand do not produce rows in this flattened landmark table. Keep a separate frame-level table when missing-detection frames must remain explicit.

### Analysis Considerations

- Treat landmarks as model predictions rather than ground-truth anatomical measurements.
- Preserve frames with empty `hands` arrays when maintaining temporal alignment.
- Do not use `hand_index` as a cross-frame tracking identifier.
- Consider handedness confidence when separating left-hand and right-hand trajectories.
- Apply temporal smoothing carefully so that it does not hide genuine rapid motion.
- Account for differences in resolution, frame rate, duration, camera placement, and recording quality across videos.
- Review detection coverage before computing movement features.
- Avoid assuming that normalized image coordinates and world coordinates are directly interchangeable.
- Use participant-level data splits for model development when multiple videos may belong to the same participant.

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
