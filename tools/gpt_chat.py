import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai
from chat_utils import load_media_data_image, load_media_data_video, load_identity, load_media_data_frames, load_media_data_frames64
from typing import List, Optional
from datasets import load_dataset
import argparse


def load_image_or_video(image_or_video, model, processor):
    _type = image_or_video["type"]
    content = image_or_video["content"]
    metadata = image_or_video.get("metadata", {})

    if _type == "image":
        load_func = load_media_data_image
    elif _type == "video":
        load_func = load_media_data_video
    elif _type == "pil_image" or _type == "pil_video":
        load_func = load_identity
    elif _type == "frames":
        load_func = load_media_data_frames
    elif _type == "frames64":
        load_func = load_media_data_frames64
    else:
        raise ValueError(f"Unknown type: {_type}")
    return load_func(content, model, processor, **metadata)

class GPTWrapper():
    def __init__(self, model_path="gpt-4o-2024-08-06") -> None:
        self.model = model_path
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.processor = None
        self.patch_size = None
        
    def __call__(self, inputs: List[dict], generation_config: Optional[dict] = None) -> str:
        images = [x for x in inputs if x["type"] in ["image", "video", "frames", "frames64", "pil_image", "pil_video"]]
        assert len(images) == 1, "Only support one image or video input"
        images = images[0]

        text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
        if "<video> " in text_prompt:
            text_prompt = text_prompt.replace("<video> ", "")
        elif "<image> " in text_prompt:
            text_prompt = text_prompt.replace("<image> ", "")

        input_images = load_image_or_video(images, self.patch_size, self.processor) 
        if images["type"] == "video": #[[Images]]
            input_images = input_images[0]

        content_blocks = []
        for img in input_images:
            content_blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}",
                    "detail": "low",
                },
            })
        content_blocks.append({"type": "text", "text": text_prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content_blocks}],
            temperature=0,
            max_tokens=1024,
        )
        generated_text = response.choices[0].message.content.strip()
        if generated_text is None:
            print(f'Error time pausing for 60 seconds: {response}')
            time.sleep(60)
            raise ValueError()

        return generated_text

    
def inference_worker(item, model, video_root, frames_root, num_frames, max_retries=10, using_frames=False):
    video_name = item["video"]
    options = ' '.join(item["options"])
    if not using_frames:
        video_path = os.path.join(video_root, video_name)
    else:
        video_path = os.path.join(frames_root, os.path.splitext(video_name)[0])


    ori_question = item["question"]
    textqa_question = ori_question + ' Keep the answer short and concise.'
    mcq_question = '\n'.join([
        'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option, with no text around it.',
        ori_question, options
    ])

    video_input = {
        "type": "video" if not using_frames else "frames64",
        "content": video_path,
        "metadata": {
            "video_num_frames": num_frames,
            "video_sample_type": "rand",
            "img_shortest_edge": 256,
            "img_longest_edge": 480,
            "max_img_seq_len": 16000,
            "do_resize": False,
        }
    }

    item["textqa_answer"] = "PENDING"
    item["mcq_answer"] = "PENDING"

    for attempt in range(max_retries):
        try:
            textqa_input = [video_input, {"type": "text", "content": f"<video> {textqa_question}"}]
            mcq_input = [video_input, {"type": "text", "content": f"<video> {mcq_question}"}]

            textqa_response = model(textqa_input)
            mcq_response = model(mcq_input)

            item["textqa_answer"] = textqa_response
            item["mcq_answer"] = mcq_response
            break
        except Exception as e:
            print(f"[Attempt {attempt+1}/{max_retries}] Error on {video_name}: {e}")
            if attempt == max_retries - 1:
                item["textqa_answer"] = "ERROR"
                item["mcq_answer"] = "ERROR"

    return item

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GPT inference on VideoEval-Pro dataset')
    parser.add_argument('--video_root', type=str, default="./videos",
                        help='Path to video files')
    parser.add_argument('--frames_root', type=str, default="./frames",
                        help='Path to video frames')
    parser.add_argument('--output_path', type=str, default="./test_results.jsonl",
                        help='Path to save output results')
    parser.add_argument('--using_frames', default=False,
                        help='Whether to use pre-extracted frames')
    parser.add_argument('--model_path', type=str, default="gpt-4o-mini-2024-07-18",
                        choices=["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gpt-4.1-2025-04-14"],
                        help='Path to GPT model')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='Number of frames to sample from video')
    parser.add_argument('--max_retries', type=int, default=10,
                        help='Maximum number of retries for failed inference')
    parser.add_argument('--num_threads', type=int, default=2,
                        help='Number of threads for parallel processing')
    
    args = parser.parse_args()

    # Load dataset from HuggingFace
    dataset = load_dataset("TIGER-Lab/VideoEval-Pro", split="test")
    
    # Convert dataset to list of items
    all_items = []
    for item in dataset:
        all_items.append({
            "video": item["video"],
            "question": item["question"],
            "options": item["options"],
            "answer": item["answer"],
            "answer_text": item["answer_text"],
            "source": item["source"],
            "qa_type": item["qa_type"],
            "qa_subtype": item["qa_subtype"],
            "meta": item["meta"]
        })

    model = GPTWrapper(model_path=args.model_path)

    processed_lines = 0
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            processed_lines = sum(1 for _ in f)

    # Skip already processed items
    all_items = all_items[processed_lines:]

    with open(args.output_path, "a", encoding="utf-8") as outfile, ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for item in all_items:
            futures.append(executor.submit(inference_worker, item, model, args.video_root, args.frames_root, 
                                         args.num_frames, args.max_retries, args.using_frames))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", dynamic_ncols=True):
            result_item = future.result()
            outfile.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            outfile.flush()

    print(f"Saved to {args.output_path}")