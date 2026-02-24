model_configs = {
            "TimeSformer": {
                "model_name": "facebook/timesformer-base-finetuned-k400",
                "num_frames": 32,
                "image_size": 224,
        },
            "VideoMAE": {
                "model_name": "MCG-NJU/videomae-base",
                "num_frames": 16,
                "image_size": 224
        },
            "ViViT": {
                "model_name": "google/vivit-b-16x2-kinetics400",
                "num_frames": 32,
                "image_size": 224
        },
        # --- NEW ---
        "VJEPA2": {
                "model_name": "facebook/vjepa2-vitg-fpc64-384-ssv2",
                "num_frames": 64,     # fpc64 expects 64 frames
                "image_size": 384,    # -256 checkpoint expects ~256 crop/height
        },
        "VJEPA2": {
                "model_name": "facebook/vjepa2-vitg-fpc64-256",
                "num_frames": 32,     # fpc64 expects 64 frames
                "image_size": 256,    # -256 checkpoint expects 256x256 images
        },
            "VideoMAEv2": {
                "model_name": "OpenGVLab/VideoMAEv2-Large",
                "num_frames": 16,     # v2 checkpoints are 16f pretrain; keep 16 for no-finetune use
                "image_size": 224     # standard crop for v2 Hugging Face release
        },
            "InternVideo2": {
                "model_name": "OpenGVLab/InternVideo2-Stage1-1B-224p-K400",
                "num_frames": 16,
                "image_size": 224,     # 224p model     
        },
    }

