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


def extract_hands_to_json(
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

    duration_sec = (
        frame_count_reported / fps
        if fps > 0 and frame_count_reported > 0
        else None
    )

    # output = {
    #     "metadata": {
    #         "video_path": video_path,
    #         "model_path": model_path,
    #         "fps": fps,
    #         "width": width,
    #         "height": height,
    #         "duration_sec": duration_sec,
    #         "frame_count_reported_by_opencv": frame_count_reported,
    #         "frame_count_processed": 0,
    #         "max_num_hands": max_num_hands,
    #         "mediapipe_model": "MediaPipe HandLandmarker Tasks API",
    #         "running_mode": "VIDEO",
    #         "min_hand_detection_confidence": min_hand_detection_confidence,
    #         "min_hand_presence_confidence": min_hand_presence_confidence,
    #         "min_tracking_confidence": min_tracking_confidence,
    #         "landmark_format": {
    #             "image_landmarks": {
    #                 "x": "normalized by image width",
    #                 "y": "normalized by image height",
    #                 "z": "relative depth; smaller values are closer to the camera",
    #             },
    #             "world_landmarks": {
    #                 "x": "real-world 3D x coordinate in meters",
    #                 "y": "real-world 3D y coordinate in meters",
    #                 "z": "real-world 3D z coordinate in meters",
    #             },
    #             "pixel_landmarks": {
    #                 "x_px": "x coordinate in pixels",
    #                 "y_px": "y coordinate in pixels",
    #             },
    #         },
    #     },
    #     "frames": [],
    # }

    output = {
        "metadata": {
            "video_name": os.path.basename(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
            "frame_count_reported_by_opencv": frame_count_reported,
            "frame_count_processed": 0,
            "max_num_hands": max_num_hands,
            "mediapipe_model": "MediaPipe HandLandmarker Tasks API",
            "running_mode": "VIDEO",
            "min_hand_detection_confidence": min_hand_detection_confidence,
            "min_hand_presence_confidence": min_hand_presence_confidence,
            "min_tracking_confidence": min_tracking_confidence,
            "landmark_format": {
                "image_landmarks": {
                    "x": "normalized by image width",
                    "y": "normalized by image height",
                    "z": "relative depth; smaller values are closer to the camera",
                },
                "world_landmarks": {
                    "x": "real-world 3D x coordinate in meters",
                    "y": "real-world 3D y coordinate in meters",
                    "z": "real-world 3D z coordinate in meters",
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
                timestamp_ms = int(round(frame_index * 1000.0 / fps))
            else:
                timestamp_ms = frame_index

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_record = {
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "hands": [],
            }

            num_detected_hands = len(result.hand_landmarks)

            for hand_idx in range(num_detected_hands):
                hand_record = {
                    "hand_index": hand_idx,
                    "handedness": None,
                    "handedness_score": None,
                    "image_landmarks": [],
                    "world_landmarks": [],
                }

                if result.handedness and hand_idx < len(result.handedness):
                    if len(result.handedness[hand_idx]) > 0:
                        cat = result.handedness[hand_idx][0]
                        hand_record["handedness"] = category_name(cat)
                        hand_record["handedness_score"] = float(cat.score)

                image_landmarks = result.hand_landmarks[hand_idx]

                for landmark_index, landmark in enumerate(image_landmarks):
                    x = float(landmark.x)
                    y = float(landmark.y)
                    z = float(landmark.z)

                    hand_record["image_landmarks"].append(
                        {
                            "landmark_index": landmark_index,
                            "x": x,
                            "y": y,
                            "z": z,
                            "x_px": x * width,
                            "y_px": y * height,
                        }
                    )

                if (
                    result.hand_world_landmarks
                    and hand_idx < len(result.hand_world_landmarks)
                ):
                    world_landmarks = result.hand_world_landmarks[hand_idx]

                    for landmark_index, landmark in enumerate(world_landmarks):
                        hand_record["world_landmarks"].append(
                            {
                                "landmark_index": landmark_index,
                                "x": float(landmark.x),
                                "y": float(landmark.y),
                                "z": float(landmark.z),
                            }
                        )

                frame_record["hands"].append(hand_record)

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

    print(f"Saved MediaPipe hand keypoints to: {output_path}")
    print(f"Processed frames: {len(output['frames'])}")