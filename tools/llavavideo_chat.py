import os
import torch
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from transformers.utils import is_flash_attn_2_available
import numpy as np
import argparse
import copy
import warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from tools.chat_utils import load_media_data_image, load_media_data_video, load_identity, load_media_data_frames

def pil_list_to_nparray(video_pil_list):
    assert all(img.size == video_pil_list[0].size for img in video_pil_list), "All images must have the same size"
    video_np = np.stack([np.array(img.convert("RGB")) for img in video_pil_list], axis=0)
    return video_np

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
    else:
        raise ValueError(f"Unknown type: {_type}")
    return load_func(content, model, processor, **metadata)


class Llava_video():
    def __init__(self, model_path="lmms-lab/LLaVA-Video-7B-Qwen2", device="cuda") -> None:

        self.device = device
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        print(f"Using {attn_implementation} for attention implementation")
        
        self.model_path = model_path
        self.model_name = "llava_qwen"

        self.tokenizer, self.model, self.image_processor, max_length = load_pretrained_model(self.model_path, None, self.model_name, torch_dtype="bfloat16", device_map="auto", attn_implementation=attn_implementation) 
        self.model.eval().to(self.device)

        self.patch_size = 14

        
    def __call__(self, inputs: List[dict], generation_config: Optional[dict] = None) -> str:
        images = [x for x in inputs if x["type"] == "image" or x["type"] == "video" or x["type"] == "frames" or x["type"] == "pil_image" or x["type"] == "pil_video"]
        assert len(images) == 1, "only support 1 input image/video"
        images = images[0]

        text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
        if "<video> " in text_prompt:
            text_prompt = text_prompt.replace("<video> ", "")
        elif "<image> " in text_prompt:
            text_prompt = text_prompt.replace("<image> ", "")

        video_pil_list = load_image_or_video(images, self.patch_size, None)
        video = pil_list_to_nparray(video_pil_list)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(self.device, dtype=torch.bfloat16)
        video = [video]

        time_instruciton = f"{len(video_pil_list)} frames are uniformly sampled from the video. Please answer the following questions related to this video:"
        question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{text_prompt}"

        conv_template = "qwen_2"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        question = conv.get_prompt()

        input_ids = tokenizer_image_token(question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        cont = self.model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        
        return text_outputs

    
def inference_worker(item, model, video_root, frames_root, num_frames, using_frames=False):
    video_name = item["video"]
    options = ' '.join(item["options"])

    video_path = os.path.join(
        frames_root if using_frames else video_root,
        os.path.splitext(video_name)[0] if using_frames else video_name
    )

    ori_question = item["question"]
    textqa_question = ori_question + ' Keep the answer short and concise.'
    mcq_question = '\n'.join([
        'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option, with no text around it.',
        ori_question, options
    ])

    video_input = {
        "type": "frames" if using_frames else "video",
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

    
    textqa_input = [video_input, {"type": "text", "content": f"<video> {textqa_question}"}]
    mcq_input = [video_input, {"type": "text", "content": f"<video> {mcq_question}"}]

    item["textqa_answer"] = model(textqa_input)
    item["mcq_answer"] = model(mcq_input)
    
    return item

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LLaVA-Video inference on VideoEval-Pro dataset')
    parser.add_argument('--video_root', type=str, default="./videos",
                        help='Path to video files')
    parser.add_argument('--frames_root', type=str, default="./frames",
                        help='Path to video frames')
    parser.add_argument('--output_path', type=str, default="./test_results.jsonl",
                        help='Path to save output results')
    parser.add_argument('--using_frames', default=False,
                        help='Whether to use pre-extracted frames')
    parser.add_argument('--model_path', type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2",
                        help='Path to LLaVA-Video model')
    parser.add_argument('--device', type=str, default="cuda",
                        help='Device to run inference on')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='Number of frames to sample from video')
    parser.add_argument('--max_retries', type=int, default=10,
                        help='Maximum number of retries for failed inference')
    parser.add_argument('--num_threads', type=int, default=1,
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

    model = Llava_video(model_path=args.model_path, device=args.device)

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
                                         args.num_frames, args.using_frames))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", dynamic_ncols=True):
            result_item = future.result()
            outfile.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            outfile.flush()

    print(f"Saved to {args.output_path}")