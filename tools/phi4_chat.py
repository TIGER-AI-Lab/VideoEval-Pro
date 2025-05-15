import os
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from datasets import load_dataset
from tools.chat_utils import load_media_data_image, load_media_data_video, load_identity, load_media_data_frames
from typing import List, Optional


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

class Phi4():
    def __init__(self, model_path="microsoft/Phi-4-multimodal-instruct", device="cuda") -> None:
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype='auto',
            _attn_implementation='flash_attention_2',
        ).to(self.device)
        print("model.config._attn_implementation:", self.model.config._attn_implementation)

        self.generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

        self.user_prompt = '<|user|>'
        self.assistant_prompt = '<|assistant|>'
        self.prompt_suffix = '<|end|>'

        self.patch_size = 14
        
    def __call__(self, inputs: List[dict], generation_config: Optional[dict] = None) -> str:
        images = [x for x in inputs if x["type"] == "image" or x["type"] == "video" or x["type"] == "frames" or x["type"] == "pil_image" or x["type"] == "pil_video"]
        assert len(images) == 1, "only support 1 input image/video"
        images = images[0]
        input_images = load_image_or_video(images, self.patch_size, None)
        if images['type'] == 'video':
            input_images = input_images[0]

        placeholder = ''
        for i in range(1, len(input_images)+1):
            placeholder += f'<|image_{i}|>'

        text_prompt = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
        if "<video> " in text_prompt:
            text_prompt = text_prompt.replace("<video> ", "")
        elif "<image> " in text_prompt:
            text_prompt = text_prompt.replace("<image> ", "")
        messages = [
            {'role': 'user', 'content': placeholder + text_prompt},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, input_images, return_tensors='pt').to(self.device)

        generation_args = {
            'max_new_tokens': 1000,
            'temperature': 0.0,
            'do_sample': False,
        }

        generate_ids = self.model.generate(
            **inputs, **generation_args, generation_config=self.generation_config, num_logits_to_keep=1,
        )

        # remove input tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response

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
        "type": "video" if not using_frames else "frames",
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
    parser = argparse.ArgumentParser(description='Run Phi-4 inference on VideoEval-Pro dataset')
    parser.add_argument('--video_root', type=str, default="./videos",
                        help='Path to video files')
    parser.add_argument('--frames_root', type=str, default="./frames",
                        help='Path to video frames')
    parser.add_argument('--output_path', type=str, default="./test_results.jsonl",
                        help='Path to save output results')
    parser.add_argument('--using_frames', default=False,
                        help='Whether to use pre-extracted frames')
    parser.add_argument('--model_path', type=str, default="microsoft/Phi-4-multimodal-instruct",
                        help='Path to Phi-4 model')
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

    model = Phi4(model_path=args.model_path, device=args.device)

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

