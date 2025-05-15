from decord import VideoReader
import cv2
import PIL
import random
import torch
import numpy as np
import PIL
from PIL import Image
import os
import base64
import io
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
from typing import List, Dict

def get_resize_output_image_size(height, width, shortest_edge, longest_edge):
    """
    Get the output size of the image after resizing given a dictionary specifying the max and min sizes.

    Args:
        image (`np.ndarray`):
            Image to resize.
        size (`Dict[str, int]`):
            Size of the output image containing the keys "shortest_edge" and "longest_edge".
        input_data_format (`ChannelDimension` or `str`):
            The channel dimension format of the input image.

    Returns:
        The output size of the image after resizing.
    """
    if shortest_edge is None and longest_edge is None:
        return height, width

    min_len = shortest_edge
    max_len = longest_edge
    aspect_ratio = width / height

    if width >= height and width > max_len:
        width = max_len
        height = int(width / aspect_ratio)
    elif height > width and height > max_len:
        height = max_len
        width = int(height * aspect_ratio)
    height = max(height, min_len)
    width = max(width, min_len)
    return height, width

def load_image_from_path(image_path):
    if isinstance(image_path, Image.Image):
        image = image_path
    else:
        image = Image.open(image_path).convert('RGB')  # PIL Image
    return [image]

def encode_pil_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_frame_indices(num_frames, vlen, sample='rand', input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def load_identity(data_path, patch_size, processor, **kwargs):
    if isinstance(data_path, tuple):
        return ([data_path[0]], data_path[1])
    else:
        return [data_path]

def load_media_data_image(data_path, patch_size, processor, **kwargs):
    img_shortest_edge = kwargs.get("img_shortest_edge", None)
    img_longest_edge = kwargs.get("img_longest_edge", None)
    image = load_image_from_path(data_path)
    if img_longest_edge is not None and img_shortest_edge is not None:
        resized_image = []
        for img in image:
            height, width = get_resize_output_image_size(img.size[1], img.size[0], img_shortest_edge, img_longest_edge)
            resized_image.append(img.resize((width, height), resample=3))
        image = resized_image
    return image

def load_media_data_frames(data_path, patch_size, processor, **kwargs):
    img_shortest_edge = kwargs.get("img_shortest_edge", None)
    img_longest_edge = kwargs.get("img_longest_edge", None)
    video_num_frames = kwargs.get("video_num_frames", 16)

    frame_files = sorted([
        os.path.join(data_path, fname)
        for fname in os.listdir(data_path)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    total_frames = len(frame_files)

    if total_frames <= video_num_frames:
        selected_indices = list(range(total_frames))
    else:
        step = total_frames / video_num_frames
        selected_indices = [int(i * step) for i in range(video_num_frames)]

    images = []
    for idx in selected_indices:
        frame_path = frame_files[idx]
        img = Image.open(frame_path).convert("RGB")
        if img_longest_edge is not None and img_shortest_edge is not None:
            height, width = get_resize_output_image_size(
                img.size[1], img.size[0],
                img_shortest_edge, img_longest_edge
            )
            img = img.resize((width, height), resample=3)
        images.append(img)

    return images

def load_media_data_frames64(data_path, patch_size, processor, **kwargs):
    img_shortest_edge = kwargs.get("img_shortest_edge", None)
    img_longest_edge = kwargs.get("img_longest_edge", None)
    video_num_frames = kwargs.get("video_num_frames", 16)
    num_workers = kwargs.get("num_workers", 8)

    frame_files = sorted([
        os.path.join(data_path, fname)
        for fname in os.listdir(data_path)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    total_frames = len(frame_files)
    if total_frames <= video_num_frames:
        selected_indices = list(range(total_frames))
    else:
        step = total_frames / video_num_frames
        selected_indices = [int(i * step) for i in range(video_num_frames)]

    def process_frame(idx):
        frame_path = frame_files[idx]
        img = Image.open(frame_path).convert("RGB")
        if img_longest_edge is not None and img_shortest_edge is not None:
            height, width = get_resize_output_image_size(
                img.size[1], img.size[0],
                img_shortest_edge, img_longest_edge
            )
            img = img.resize((width, height), resample=3)
        return encode_pil_to_base64(img)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_frame, selected_indices))

    return results

def load_media_data_video(data_path, patch_size, processor, **kwargs):
    img_shortest_edge = kwargs.get("img_shortest_edge", None)
    img_longest_edge = kwargs.get("img_longest_edge", None)
    max_img_seq_len = kwargs.get("max_img_seq_len", None)
    video_num_frames = kwargs.get("video_num_frames", 16)
    video_sample_type = kwargs.get("video_sample_type", "rand")
    do_resize = kwargs.get("do_resize", False)
    model_patch_size = patch_size

    video_reader = VideoReader(data_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    if video_num_frames == 'auto':
        if not do_resize:
            vid = cv2.VideoCapture(data_path)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            vid.release()
        else:
            height = processor.image_processor.size['height']
            width = processor.image_processor.size['width']
        num_patches = int((height // model_patch_size) * (width // model_patch_size))
        video_num_frames = int(max_img_seq_len // num_patches)
    frame_indices = get_frame_indices(video_num_frames, vlen, sample=video_sample_type, input_fps=fps)
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C)
    results = []
    for frame in frames:
        img = PIL.Image.fromarray(frame, mode="RGB")
        if img_shortest_edge is not None and img_longest_edge is not None:
            height, width = get_resize_output_image_size(img.size[1], img.size[0], img_shortest_edge, img_longest_edge)
            img = img.resize((width, height), resample=3)
        results.append(img)
    
    return [results]

def load_media_data_video_kangaroo(data_path, patch_size, processor, **kwargs):
    img_shortest_edge = kwargs.get("img_shortest_edge", None)
    img_longest_edge = kwargs.get("img_longest_edge", None)
    max_img_seq_len = kwargs.get("max_img_seq_len", None)
    video_num_frames = kwargs.get("video_num_frames", 16)
    video_sample_type = kwargs.get("video_sample_type", "rand")
    do_resize = kwargs.get("do_resize", False)
    model_patch_size = 14

    video_reader = VideoReader(data_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    if video_num_frames == 'auto':
        if not do_resize:
            vid = cv2.VideoCapture(data_path)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            vid.release()
        else:
            height = 448
            width = 448
        num_patches = int((height // model_patch_size) * (width // model_patch_size))
        video_num_frames = int(max_img_seq_len // num_patches)
    frame_indices = get_frame_indices(video_num_frames, vlen, sample=video_sample_type, input_fps=fps)
    durations = [idx / fps  for idx in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C)
    results = []
    for frame in frames:
        img = PIL.Image.fromarray(frame, mode="RGB")
        if img_shortest_edge is not None and img_longest_edge is not None:
            height, width = get_resize_output_image_size(img, img_shortest_edge, img_longest_edge)
            img = img.resize((width, height), resample=3)
        results.append(img)
    
    return ([results], torch.Tensor(durations))