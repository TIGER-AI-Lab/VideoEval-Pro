import os
import torch
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import numpy as np
import re
import argparse
from PIL import Image
import cv2
import transformers
import uuid
from datasets import load_dataset
if transformers.__version__ > '4.36':
    truncate_inputs = False

from models.longllava.model import *
from models.longllava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.longllava.conversation import conv_templates
from models.longllava.model.builder import load_pretrained_model
from models.longllava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

from tools.chat_utils import load_media_data_image, load_media_data_video, load_identity, load_media_data_frames

def split_image(image, n=2):
    if n==1: return [image]
    width, height = image.size
    block_width = width // n
    block_height = height // n

    blocks = []

    for i in range(n):
        for j in range(n):
            left = j * block_width
            upper = i * block_height
            right = (j + 1) * block_width
            lower = (i + 1) * block_height
            block = image.crop((left, upper, right, lower))
            blocks.append(block)
    blocks.append(image)

    return blocks

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

class LongLLaVA_chatbot():
    def __init__(
        self, 
        model_path="/map-vepfs/weiming/checkpoints/FreedomIntelligence/LongLLaVA-9B", device="cuda:0",
        ) -> None:
        """Model wrapper
        Args:
            model_path (str): Path to the model.
            device (str): Device to use (default: "cuda:0").
        """
        
        self.gen_kwargs = {
            'do_sample': False,
            'max_new_tokens': 768,
            'min_new_tokens': 1,
            'temperature': .0,
        }
            
        self.device = device
        
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, False, False, use_flash_attn=True)
        self.model = model
        self.conv_mode = "jamba"
        self.jamba_process_images = process_images
        self.jamba_tokenizer_image_token = tokenizer_image_token
        self.truncate_input = True
        self.jamba_conv_templates = conv_templates
        eos_token_id = tokenizer.eos_token_id
        self.gen_kwargs['eos_token_id'] = eos_token_id
        self.gen_kwargs['pad_token_id'] = tokenizer.pad_token_id if tokenizer.pad_token_id else eos_token_id
        print(f'setting eos_token_id to {eos_token_id}')

        model.eval()
        self.tokenizer = tokenizer
        self.processor = image_processor
        self.history = []
        self.images = []
        self.debug = True

    def clear_history(self,):
        self.images = []
        self.history = []


    def tokenizer_image_token(self, prompt, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None): # copied from llava
        prompt_chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids


    def chat_with_jamba(self, text, images, video, patchside_length=336, patchStrategy='norm'):
       
        def insert_image_placeholder_for_video(t, num_images, placeholder='<img><image></img>', tag='<t>'):
            result = '<vid>'
            for _ in range(num_images):
                result += f"{placeholder}{tag}"
            result = result.rstrip(tag) + '</vid>'
            result = result + t
            return result

        def processForBestFitPatch(text, images, output_dir='./LongLLaVA/data/TestBestFit', patchside_length=336):
            side_length = patchside_length
            placeholder_count = text.count('<image>')
            if placeholder_count != len(images):
                raise ValueError("The number of <image> placeholders does not match the number of images.")
            
            new_image_paths = []
            os.makedirs(output_dir, exist_ok=True)

            for idx, image_path in enumerate(images):
                if isinstance(image_path, str) and os.path.isfile(image_path):
                    image = cv2.imread(image_path)
                elif isinstance(image_path, Image.Image):  # Assuming image_path is a numpy array representing an image
                    image = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
                    random_filename = str(uuid.uuid4()) + '.jpg'
                    random_path = os.path.join(output_dir, 'ori', 'images', random_filename)
                    cv2.imwrite(random_path, image)
                    image_path = random_path
                    
                if image is None:
                    raise FileNotFoundError(f"Image not found: {image_path}")

                height, width = image.shape[:2]

                new_height = ((height + side_length - 1) // side_length) * side_length
                new_width = ((width + side_length - 1) // side_length) * side_length
                pad_height = (new_height - height) // 2
                pad_width = (new_width - width) // 2

                padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[255, 255, 255])

                split_images = [image_path]
                path_parts = image_path.split('/')
                base_name = os.path.splitext(path_parts[-1])[0]
                if len(path_parts) >= 3:
                    subdir = os.path.join(output_dir, path_parts[-3], path_parts[-2])
                else:
                    subdir = os.path.join(output_dir, path_parts[-2])

                os.makedirs(subdir, exist_ok=True)

                for i in range(0, new_height, side_length):
                    for j in range(0, new_width, side_length):
                        split_img = padded_image[i:i+side_length, j:j+side_length]
                        split_path = os.path.join(subdir, f'{base_name}_{i//side_length}_{j//side_length}.jpg')
                        if not os.path.exists(split_path):
                            cv2.imwrite(split_path, split_img)
                        split_images.append(split_path)
                
                row_count = new_height // side_length
                col_count = new_width // side_length
                
                replace_str = '<image>\n' + '\n'.join(['<img>' + '</img><img>'.join(['<image>' for _ in range(col_count)]) + '</img>' for _ in range(row_count)])
                text = text.replace('<image>', replace_str, 1)
                
                new_image_paths.extend(split_images)

            final_placeholder_count = text.count('<image>')
            if final_placeholder_count != len(new_image_paths):
                print(new_image_paths)
                print(placeholder_count)
                raise ValueError("The number of processed <image> placeholders does not match the number of split images.")
            
            return text, new_image_paths

        images = images[0]
        text = insert_image_placeholder_for_video(text, len(images)) if video else text
        
        num_images_in_text = text.count('<image>')
        if num_images_in_text < len(images):
            missing_images = len(images) - num_images_in_text
            text = '<image>' * missing_images + text
        if '</img>' not in text:
                text = text.replace('<image>', '<img><image></img>')

        if len(images):
            if 'bestFit' in patchStrategy:
                text, images = processForBestFitPatch(text, images, patchside_length=patchside_length)
            elif 'norm'!=patchStrategy:
                print('Error: patchStrategy is not Impplmented')

        self.images=images

        # make conv
        conv = self.jamba_conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # make input ids
        input_ids = self.jamba_tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        if self.images != [None]:
            lenth = len(images)
            image_tensors = self.jamba_process_images(self.images, self.processor, {}).to(self.device, dtype=torch.float16)
        else:
            image_tensors = None

        output_ids = self.model.generate(
                input_ids,
                images=image_tensors,
                use_cache=True,
                **self.gen_kwargs)

        try:
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        except:
            answer = "Z"

        return answer

    def chat(self, text: str, images: None, video=True, patchside_length=336, patchStrategy='norm'):
        '''
        images: list[str], images for this round
        text: str
        '''

        return self.chat_with_jamba(text, images, video, patchside_length, patchStrategy)


class LongLLaVA():
    def __init__(self, model_path="FreedomIntelligence/LongLLaVA-9B", device="cuda") -> None:

        self.bot = LongLLaVA_chatbot(
            model_path=model_path, device=device, 
            )
        self.image_processor = self.bot.processor
        self.patch_size = 14

    def __call__(self, inputs: List[dict], generation_config: Optional[dict] = None) -> str:
        images = [x for x in inputs if x["type"] == "image" or x["type"] == "video" or x["type"] == "frames" or x["type"] == "pil_image" or x["type"] == "pil_video"]
        assert len(images) == 1, "only support 1 input image/video"
        images = images[0]

        text = "\n".join([x["content"] for x in inputs if x["type"] == "text"])
        if "<video> " in text:
            text = text.replace("<video> ", "")
        elif "<image> " in text:
            text = text.replace("<image> ", "")

        input_images = load_image_or_video(images, self.patch_size, self.image_processor)
        if images["type"] == "frames":
            input_images = [input_images]

        answer = self.bot.chat(
            text=text,
            images=input_images, 
            video=True,
            patchside_length=336, 
            patchStrategy="norm"
            )
        
        return answer

    
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
    parser = argparse.ArgumentParser(description='Run LongLLaVA inference on VideoEval-Pro dataset')
    parser.add_argument('--video_root', type=str, default="./videos",
                        help='Path to video files')
    parser.add_argument('--frames_root', type=str, default="./frames",
                        help='Path to video frames')
    parser.add_argument('--output_path', type=str, default="./test_results.jsonl",
                        help='Path to save output results')
    parser.add_argument('--using_frames', default=False,
                        help='Whether to use pre-extracted frames')
    parser.add_argument('--model_path', type=str, default="FreedomIntelligence/LongLLaVA-9B",
                        help='Path to LongLLaVA model')
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

    model = LongLLaVA(model_path=args.model_path, device=args.device)

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
