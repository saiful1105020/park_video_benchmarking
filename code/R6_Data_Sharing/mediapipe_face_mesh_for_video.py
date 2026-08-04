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


def extract_face_mesh_to_numpy(
    video_path: str,
    output_path: str,
    model_path: str,
    max_num_faces: int = 1,
    min_face_detection_confidence: float = 0.5,
    min_face_presence_confidence: float = 0.5,
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
    all_face_landmarks = []
    timestamps_ms = []

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=max_num_faces,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frame_index = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
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

            # Determine number of landmarks (468 standard, 478 with refined irises)
            num_landmarks = (
                len(result.face_landmarks[0]) if result.face_landmarks else 478
            )

            # Initialize zero buffer for current frame: (max_num_faces, num_landmarks, 3)
            face_lmks_frame = np.zeros((max_num_faces, num_landmarks, 3), dtype=np.float32)

            num_detected_faces = len(result.face_landmarks)

            for face_idx in range(min(num_detected_faces, max_num_faces)):
                image_landmarks = result.face_landmarks[face_idx]
                for lm_idx, lm in enumerate(image_landmarks):
                    face_lmks_frame[face_idx, lm_idx] = [lm.x, lm.y, lm.z]

            all_face_landmarks.append(face_lmks_frame)
            frame_index += 1

    cap.release()

    frame_count_processed = len(all_face_landmarks)
    duration_sec = (
        frame_count_processed / fps if fps > 0 else 0.0
    )

    # Convert to single contiguous array: Shape (frames, max_num_faces, 478, 3)
    np_face_landmarks = np.array(all_face_landmarks, dtype=np.float32)
    np_timestamps = np.array(timestamps_ms, dtype=np.int64)

    metadata = {
        "video_name": os.path.basename(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
        "frame_count_reported_by_opencv": frame_count_reported,
        "frame_count_processed": frame_count_processed,
        "max_num_faces": max_num_faces,
    }

    output_path = os.path.abspath(output_path)
    if not output_path.endswith(".npz") and not output_path.endswith(".npy"):
        output_path += ".npz"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save uncompressed NumPy archive (.npz) containing array & metadata
    np.savez(
        output_path,
        face_landmarks=np_face_landmarks,
        timestamps_ms=np_timestamps,
        metadata=np.array(metadata, dtype=object),
    )

    print(f"Saved uncompressed NumPy archive to: {output_path}")
    print(f"Processed frames: {frame_count_processed}")
    print(f"Tensors shape: {np_face_landmarks.shape}")