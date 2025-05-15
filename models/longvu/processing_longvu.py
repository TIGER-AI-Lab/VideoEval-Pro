"""
Processor class for LongVA.
"""

import os
import json
from typing import List, Optional, Union, Dict

from transformers.feature_extraction_sequence_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType
from transformers.processing_utils import transformers_module
from transformers.utils import IMAGE_PROCESSOR_NAME

from PIL import Image
import logging
import torch
import numpy as np

from models.longvu.constants import IMAGE_TOKEN_INDEX
from models.longvu.mm_datautils import tokenizer_image_token, process_images

logger = logging.getLogger(__name__)


class LongVUProcessor:
    def __init__(self, image_processor=None, tokenizer=None, model_config=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        do_resize=True,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        add_image_ids: bool = True,
    ) -> BatchFeature:
        assert isinstance(text, str), "text must be a string"
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            mode = "image"
        elif isinstance(images, list) and isinstance(images[0], list):
            mode = "video"
        else:
            raise ValueError("Invalid input images. images must be a PIL image or a list of PIL images (images), or a list of list of PIL images (videos).")

        if "<image> " in text:
            text = text.replace("<image> ", "<image>\n", 1)
        input_ids = tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        # text_inputs: 
        # 1. input_ids: [batch_size, sequence_length], e.g. [1, 6]
        # 2. attention_mask: [batch_size, sequence_length], e.g. [1, 6]
        
        # check the number of image token ids, and truncated the number of images if needed
        pixel_values = None
        image_sizes = None
        if images is not None:
            if mode == "image":
                pixel_values = process_images(images, self.image_processor, self.model_config)
                image_sizes = images[0].size
            elif mode == "video": # video
                # flatten images
                images = images[0]
                videos = []
                for video in images:
                    np_video = np.array(video)
                    videos.append(np_video)
                videos = np.stack(videos)
                image_sizes = videos[0].shape[:2]
                videos = process_images(videos, self.image_processor, self.model_config)
                pixel_values = [item.unsqueeze(0) for item in videos]
        return BatchFeature(data={"input_ids": input_ids, "attention_mask": attention_mask, "images": pixel_values, "modalities": mode, "image_sizes": image_sizes})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)