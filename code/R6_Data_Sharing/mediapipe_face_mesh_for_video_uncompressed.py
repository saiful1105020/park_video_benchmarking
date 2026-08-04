#!/usr/bin/env python3

import argparse
import json
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


def extract_face_mesh_to_json(
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

    duration_sec = (
        frame_count_reported / fps
        if fps > 0 and frame_count_reported > 0
        else None
    )

    output = {
        "metadata": {
            "video_name": os.path.basename(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
            "frame_count_reported_by_opencv": frame_count_reported,
            "frame_count_processed": 0,
            "max_num_faces": max_num_faces,
            "mediapipe_model": "MediaPipe FaceLandmarker Tasks API",
            "running_mode": "VIDEO",
            "min_face_detection_confidence": min_face_detection_confidence,
            "min_face_presence_confidence": min_face_presence_confidence,
            "min_tracking_confidence": min_tracking_confidence,
            "landmark_format": {
                "image_landmarks": {
                    "x": "normalized by image width",
                    "y": "normalized by image height",
                    "z": "relative depth; smaller values are closer to the camera",
                },
                "pixel_landmarks": {
                    "x_px": "x coordinate in pixels",
                    "y_px": "y coordinate in pixels",
                },
            },
        },
        "frames": [],
    }

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
                timestamp_ms = int(round(frame_index * 1000.0 / fps))
            else:
                timestamp_ms = frame_index

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            
            frame_record = {
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "faces": [],
            }

            num_detected_faces = len(result.face_landmarks)

            for face_idx in range(num_detected_faces):
                face_record = {
                    "face_index": face_idx,
                    "image_landmarks": []
                }

                image_landmarks = result.face_landmarks[face_idx]
                
                for landmark_index, landmark in enumerate(image_landmarks):
                    
                    x = float(landmark.x)
                    y = float(landmark.y)
                    z = float(landmark.z)

                    face_record["image_landmarks"].append(
                        {
                            "landmark_index": landmark_index,
                            "x": x,
                            "y": y,
                            "z": z,
                            "x_px": x * width,
                            "y_px": y * height,
                        }
                    )

                frame_record["faces"].append(face_record)
                

            output["frames"].append(frame_record)
            frame_index += 1

    cap.release()

    output["metadata"]["frame_count_processed"] = len(output["frames"])

    if output["metadata"]["duration_sec"] is None and fps > 0:
        output["metadata"]["duration_sec"] = len(output["frames"]) / fps

    output_path = os.path.abspath(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved MediaPipe face keypoints to: {output_path}")
    print(f"Processed frames: {len(output['frames'])}")