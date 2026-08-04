#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def category_name(category):
    """Handle small naming differences across MediaPipe versions."""
    return (
        getattr(category, "category_name", None)
        or getattr(category, "categoryName", None)
        or getattr(category, "display_name", None)
    )


def extract_hands_to_numpy(
    video_path: str,
    output_path: str,
    model_path: str,
    max_num_hands: int = 2,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    video_path = os.path.abspath(video_path)
    model_path = os.path.abspath(model_path)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MediaPipe model file not found: {model_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pre-allocate frames list for accumulation
    all_image_landmarks = []
    all_world_landmarks = []
    all_handedness = []
    all_scores = []
    timestamps_ms = []

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frame_index = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame_bgr = cap.read()

            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb,
            )

            if fps > 0:
                ts_ms = int(round(frame_index * 1000.0 / fps))
            else:
                ts_ms = frame_index

            result = landmarker.detect_for_video(mp_image, ts_ms)
            timestamps_ms.append(ts_ms)

            # Initialize zero buffers for the frame: (max_num_hands, 21, 3)
            img_lmks_frame = np.zeros((max_num_hands, 21, 3), dtype=np.float32)
            world_lmks_frame = np.zeros((max_num_hands, 21, 3), dtype=np.float32)
            handedness_frame = np.zeros((max_num_hands,), dtype=np.int8)  # 0: None, 1: Left, 2: Right
            scores_frame = np.zeros((max_num_hands,), dtype=np.float32)

            num_detected_hands = len(result.hand_landmarks)

            for hand_idx in range(min(num_detected_hands, max_num_hands)):
                # Parse handedness
                if result.handedness and hand_idx < len(result.handedness):
                    if len(result.handedness[hand_idx]) > 0:
                        cat = result.handedness[hand_idx][0]
                        label = category_name(cat)
                        scores_frame[hand_idx] = float(cat.score)
                        if label == "Left":
                            handedness_frame[hand_idx] = 1
                        elif label == "Right":
                            handedness_frame[hand_idx] = 2

                # Parse normalized image landmarks (21 x 3)
                image_landmarks = result.hand_landmarks[hand_idx]
                for lm_idx, lm in enumerate(image_landmarks):
                    img_lmks_frame[hand_idx, lm_idx] = [lm.x, lm.y, lm.z]

                # Parse real-world 3D landmarks (21 x 3)
                if (
                    result.hand_world_landmarks
                    and hand_idx < len(result.hand_world_landmarks)
                ):
                    world_landmarks = result.hand_world_landmarks[hand_idx]
                    for lm_idx, lm in enumerate(world_landmarks):
                        world_lmks_frame[hand_idx, lm_idx] = [lm.x, lm.y, lm.z]

            all_image_landmarks.append(img_lmks_frame)
            all_world_landmarks.append(world_lmks_frame)
            all_handedness.append(handedness_frame)
            all_scores.append(scores_frame)

            frame_index += 1

    cap.release()

    frame_count_processed = len(all_image_landmarks)
    duration_sec = (
        frame_count_processed / fps if fps > 0 else 0.0
    )

    # Convert to NumPy arrays
    # Shape: (frames, max_num_hands, 21, 3)
    np_image_landmarks = np.array(all_image_landmarks, dtype=np.float32)
    np_world_landmarks = np.array(all_world_landmarks, dtype=np.float32)
    
    # Shape: (frames, max_num_hands)
    np_handedness = np.array(all_handedness, dtype=np.int8)
    np_scores = np.array(all_scores, dtype=np.float32)
    np_timestamps = np.array(timestamps_ms, dtype=np.int64)

    metadata = {
        "video_name": os.path.basename(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
        "frame_count_reported_by_opencv": frame_count_reported,
        "frame_count_processed": frame_count_processed,
        "max_num_hands": max_num_hands,
        "handedness_map": {0: "None", 1: "Left", 2: "Right"},
    }

    output_path = os.path.abspath(output_path)
    if not output_path.endswith(".npz") and not output_path.endswith(".npy"):
        output_path += ".npz"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save uncompressed NumPy archive (.npz) containing tensors & metadata
    np.savez(
        output_path,
        image_landmarks=np_image_landmarks,
        world_landmarks=np_world_landmarks,
        handedness=np_handedness,
        handedness_scores=np_scores,
        timestamps_ms=np_timestamps,
        metadata=np.array(metadata, dtype=object),
    )

    print(f"Saved uncompressed NumPy archive to: {output_path}")
    print(f"Processed frames: {frame_count_processed}")
    print(f"Tensors shape: {np_image_landmarks.shape}")
    print(f"World landmarks shape: {np_world_landmarks.shape}")
    print(f"Handedness shape: {np_handedness.shape}")
    print(f"Scores shape: {np_scores.shape}")
    print(f"Timestamps shape: {np_timestamps.shape}")