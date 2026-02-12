### Video Foundation Model Configurations and Sizes

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