import time
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from filehash import get_file_hash
from helper import get_available_device
from logger import Logger


class Whisper:
    """
    A class to handle Whisper speech recognition tasks
    """

    def __init__(
        self,
        *,
        input_path: str,
    ):
        """
        Initialize Whisper model and processor

        Args:
            model_size (str): Size of the model ('tiny', 'base', 'small', 'medium', 'large')
            device (str): Device to run the model on ('cuda', 'cpu', or None for auto-detection)
        """
        # Init logger
        self.logger = Logger.setup(__name__)

        # Set whisper state
        self.input_path = Path(input_path)
        self.input_file_hash = get_file_hash(self.input_path)

        # Set device
        self.model_id = "openai/whisper-large-v3-turbo"
        self.device = get_available_device()
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Initialize model and processor
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            # Apply optimizations
            if self.device != "mps":  # torch.compile not supported on MPS
                self.model.generation_config.cache_implementation = "static"
                self.model.generation_config.max_new_tokens = 256
                self.model.forward = torch.compile(
                    self.model.forward, mode="reduce-overhead", fullgraph=True
                )

            self.model = self.model.to(self.device)

            # Clear any existing forced_decoder_ids to avoid conflicts with language setting
            self.model.generation_config.forced_decoder_ids = None
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {str(e)}")
            raise

    def transcribe_audio(self, audio_path: Path) -> list[dict[str, str]]:
        """
        Transcribe audio data with timestamps

        Args:
            audio (numpy.ndarray): Audio data
            sampling_rate (int): Sampling rate of the audio
            **kwargs: Additional arguments to pass to the generation step

        Returns:
            A list of dicts with "timestamp" and "text" keys i.e.
            [{"timestamp": (0.0, 4.0), "text": "Hello"}]
        """
        start = time.perf_counter()
        self.logger.info(f"Transcribing audio from {audio_path}")
        try:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                framework="pt",
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                model_kwargs={"use_cache": True},
            )

            # Hard code english language assumption for speed
            result = pipe(
                str(audio_path),
                return_timestamps=True,
                generate_kwargs={
                    "use_cache": True,
                    "task": "transcribe",
                    "language": "en",  # Explicitly set language
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to transcribe audio: {str(e)}")
            raise
        end = time.perf_counter()
        self.logger.info(
            "Transcription completed in {:.4f} seconds".format(end - start)
        )
        return result["chunks"]  # type: ignore
