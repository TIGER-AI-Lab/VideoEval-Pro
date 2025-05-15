# VideoEval-Pro
This repository contains the evaluation code for the VideoEval-Pro. 
The data is available on HuggingFace: [VideoEval-Pro](https://huggingface.co/datasets/TIGER-Lab/VideoEval-Pro)

## Dataset Introduction
VideoEval-Pro is a robust and realistic long video understanding benchmark containing open-ended, short-answer QA problems. The dataset is constructed by reformatting questions from four existing long video understanding MCQ benchmarks: Video-MME, MLVU, LVBench, and LongVideoBench into free-form questions.




Each example in the dataset contains:
- `video`: Name (path) of the video file
- `question`: The question about the video content
- `options`: Original options from the source benchmark
- `answer`: The correct MCQ answer
- `answer_text`: The correct free-form answer
- `meta`: Additional metadata from the source benchmark
- `source`: Source benchmark
- `qa_subtype`: Question task subtype
- `qa_type`: Question task type

## Evaluation Steps

1. **Download and Prepare Videos**
   ```bash
   # Download the dataset from HuggingFace
   git lfs install
   git clone https://huggingface.co/datasets/TIGER-Lab/VideoEval-Pro

   # Navigate to videos directory
   cd VideoEval-Pro/videos
   
   # Merge all split tar.gz files into a single archive
   cat videos_part_*.tar.gz > videos_merged.tar.gz
   
   # Extract the merged archive
   tar -xzf videos_merged.tar.gz
   
   # [Optional] Clean up the split files and merged archive
   rm videos_part_*.tar.gz videos_merged.tar.gz
   
   # After extraction, you will get a directory containing all videos
   # The path to this directory will be used as --video_root in evaluation
   # For example: 'VideoEval-Pro/videos'
   ```

2. **[Optional] Pre-extract Frames**
   To improve efficiency, you can pre-extract frames from videos. The extracted frames should be organized as follows:
   ```
   frames_root/
   ├── video_name_1/              # Video name
   │   ├── 000001.jpg             # Frame images
   │   ├── 000002.jpg
   │   └── ...
   ├── video_name_2/
   │   ├── 000001.jpg
   │   ├── 000002.jpg
   │   └── ...
   └── ...
   ```

   After frame extraction, the path to the frames will be used as `--frames_root`. Set `--using_frames True` when running the evaluation script.

3. **Setup Evaluation Environment**
   ```bash
   # Clone the repository from the GitHub repository
   git clone https://github.com/TIGER-AI-Lab/VideoEval-Pro
   cd VideoEval-Pro
   
   # Create conda environment from requirements.txt (there are different env files for different models)
   conda create -n videoevalpro --file *.yaml
   conda activate videoevalpro
   ```

4. **Run Evaluation**
   ```bash
   cd VideoEval-Pro
   
   # Set PYTHONPATH
   export PYTHONPATH=.
   
   # Run evaluation script with the following parameters:
   # --video_root: Path to video files folder
   # --frames_root: Path to video frames folder [For using_frames]
   # --output_path: Path to save output results
   # --using_frames: Whether to use pre-extracted frames
   # --model_path: Path to model
   # --device: Device to run inference on
   # --num_frames: Number of frames to sample from video
   # --max_retries: Maximum number of retries for failed inference
   # --num_threads: Number of threads for parallel processing
   
   python tools/*_chat.py \
       --video_root <path_to_videos> \
       --frames_root <path_to_frames> \
       --output_path <path_to_save_results> \
       --using_frames <True/False> \
       --model_path <model_name_or_path> \
       --device <device> \
       --num_frames <number_of_frames> \
       --max_retries <max_retries> \
       --num_threads <num_threads>

   E.g.:
   python tools/qwen_chat.py \
       --video_root ./videos \
       --frames_root ./frames \
       --output_path ./results/qwen_results.jsonl \
       --using_frames False \
       --model_path Qwen/Qwen2-VL-7B-Instruct \
       --device cuda \
       --num_frames 32 \
       --max_retries 10 \
       --num_threads 1
   ```