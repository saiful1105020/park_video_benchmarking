import numpy as np

data = np.load("/localdisk1/PARK/park_video_benchmarking/code/R6_Data_Sharing/Sample_Data/NIHZY217YWJA8-reverse_count-2021-03-17T18-17-02-770Z--face-mesh-mediapipe.npz", allow_pickle=True)

# Read Metadata
metadata = data["metadata"].item()
print(metadata)

# Read face landmarks
face_lmks = data["face_landmarks"]  # Shape: (frames, max_faces, 478, 3)

# Get pixel coordinates for a specific frame and face
frame_index = 10
face_index = 0
x_pixels = face_lmks[frame_index, face_index, :, 0] * metadata["width"]
y_pixels = face_lmks[frame_index, face_index, :, 1] * metadata["height"]

# Shape is (478,) -- coordinate for each face landmark
print(x_pixels.shape)
print(y_pixels.shape)

# Print the coordinates
print(x_pixels)
print(y_pixels)