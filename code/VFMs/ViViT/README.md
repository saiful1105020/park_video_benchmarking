### Frozen Embedding Extraction for ViViT (Batch Processing)

The ```requirements.txt``` file lists the Python environment necessary to extract frozen embeddings for the TimeSformer model.

The script ```extract_frozen_embeddings.py``` contains a working demo for frozen embedding extraction. Update these three lines inside the ```main()``` function:

```python
    # When the batch is really long, interruption can happen. So, we want to keep track of progress (how many videos are processed already)
    count_filename = f'/localdisk1/{project_dir}/{project_name}/code/VFMs/{model_tag}/{model_tag}_Features_Completed_Count.txt'
    
    # This file will contain the embeddings (list of dictionary format)
    embeddings_filename = f'/localdisk1/{project_dir}/{project_name}/code/VFMs/{model_tag}_Features_All_Videos.pkl'

    # This is the input directory. All videos inside this directory will be processed in a batch.
    input_video_dir = f'/localdisk1/{project_dir}/{project_name}/code/VFMs/sample_data'
```

A sample video is provided (downloaded from YouTube) ```../sample_data/sample_youtube_video.mp4```. Note that this does not represent any video from our dataset (we are unable to share raw videos to comply with patient privacy).
