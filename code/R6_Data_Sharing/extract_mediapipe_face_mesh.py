# PID: 3125423
import os
import pandas as pd
# from park_video_benchmarking.code.R6_Data_Sharing.mediapipe_face_mesh_for_video_uncompressed import extract_face_mesh_to_json
from mediapipe_face_mesh_for_video import extract_face_mesh_to_numpy

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

SAVE_DIR = "/localdisk3/park_video_benchmarking_additional_data/"
LOCAL_DIR = "mediapipe_face_mesh_compressed"
RAW_DATA_DIR = "/localdisk1/PARK/park_video_benchmarking/data/videos/raw_videos"
MODEL_PATH = "/localdisk1/PARK/park_video_benchmarking/code/R6_Data_Sharing/face_landmarker_v2_with_blendshapes.task"

max_num_faces = 1
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
min_presence_confidence = 0.5

# save the list of processed videos here
PROCESSED_VIDEOS_TRACKING_FILE = os.path.join(SAVE_DIR, LOCAL_DIR, "processed_videos.csv")
if not os.path.exists(PROCESSED_VIDEOS_TRACKING_FILE):
    os.makedirs(os.path.dirname(PROCESSED_VIDEOS_TRACKING_FILE), exist_ok=True)

# read already processed videos if the file exists
if os.path.exists(PROCESSED_VIDEOS_TRACKING_FILE):
    processed_videos = pd.read_csv(PROCESSED_VIDEOS_TRACKING_FILE)["video_name"].tolist()
else:
    processed_videos = []

if __name__ == "__main__":
    # check how many videos are in the raw data directory
    raw_videos = [x for x in os.listdir(RAW_DATA_DIR) if x.endswith(".mp4")]
    print(f"Number of raw videos: {len(raw_videos)}")

    count_processed = 0
    for video in raw_videos:
        if video in processed_videos:
            print(f"Video {video} has already been processed. Skipping.")
            continue

        try:
            # keypoints_file  = video.replace(".mp4", "-face-mesh-mediapipe.json")
            keypoints_file  = video.replace(".mp4", "-face-mesh-mediapipe.npz")
            output_path = os.path.join(SAVE_DIR, LOCAL_DIR, keypoints_file)
            video_path = os.path.join(RAW_DATA_DIR, video)

            if not os.path.exists(video_path):
                print(f"Video {video} does not exist in the raw data directory.")

            extract_face_mesh_to_numpy(
                video_path=video_path,
                output_path=output_path,
                model_path=MODEL_PATH,
                max_num_faces=max_num_faces,
                min_face_detection_confidence=min_detection_confidence, # note that this is for face detection, not hand detection
                min_face_presence_confidence=min_presence_confidence,    # note that this is for face presence, not hand presence
                min_tracking_confidence=min_tracking_confidence,
            )

            # add the video to the list of processed videos
            processed_videos.append(video)
            count_processed += 1
        except Exception as e:
            print(f"Error processing video {video}: {e}")

        # exit(0)

        if count_processed % 10 == 0:
            print(f"Processed {count_processed} videos so far.")
            # save the list of processed videos
            pd.DataFrame({"video_name": processed_videos}).to_csv(PROCESSED_VIDEOS_TRACKING_FILE, index=False)

    print(f"Finished processing videos. Total processed in this batch: {count_processed}")