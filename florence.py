from pathlib import Path
from typing import Literal, Union

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from filehash import get_file_hash
from helper import get_available_device
from logger import Logger


class Florence:
    """
    A class to handle video model tasks
    """

    def __init__(self, input_path: str):
        self.logger = Logger.setup(__name__)

        self.input_path = Path(input_path)
        self.input_file_hash = get_file_hash(self.input_path)

        self.model_id = "microsoft/Florence-2-base"
        self.device = get_available_device()
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
        except Exception as e:
            self.logger.error(f"Failed to load Florence model: {str(e)}")
            raise

    def get_video_caption(
        self,
        image_path: str,
        prompt: Union[
            Literal["<CAPTION>"],
            Literal["<DETAILED_CAPTION>"],
            Literal["<MORE_DETAILED_CAPTION>"],
        ] = "<MORE_DETAILED_CAPTION>",
    ):
        """
        Generate a caption for the provided video.

        Args:
            video_path (str): Path to the video file

        Returns:
            str: The generated caption
        """
        image = Image.open(image_path)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )

        return parsed_answer
