import logging
import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Union
import time

from tqdm import tqdm

from filehash import get_file_hash
from helper import get_filenames_of_dir
from logger import Logger


class FFmpeg:
    """Production-ready FFmpeg wrapper with simplified interface"""

    def __init__(
        self, *, input_path: str, ffmpeg_path: str, tmp_path: str, timeout: int = 3600
    ):
        self.log_level = logging.INFO
        self.logger = Logger.setup(__name__, self.log_level)

        self.timeout = timeout
        self.input_path = Path(input_path)
        self.ffmpeg_path = Path(ffmpeg_path)
        self.tmp_path = Path(tmp_path)

        self.file_hash = get_file_hash(self.input_path)

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        if not shutil.which(self.ffmpeg_path):
            raise FileNotFoundError(f"FFmpeg not found at path: {self.ffmpeg_path}")

        if not self.tmp_path.exists() or not self.tmp_path.is_dir():
            raise FileNotFoundError(f"Temporary directory not found: {self.tmp_path}")

    def _get_ffmpeg_loglevel(self) -> str:
        """
        Convert Python logging level to FFmpeg loglevel.
        """
        return logging.getLevelName(self.log_level).lower()

    def _run_command(
        self,
        args: List[str],
        input_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> subprocess.CompletedProcess:
        """Execute FFmpeg command with proper error handling and logging"""

        cmd_list = [
            str(self.ffmpeg_path),
            "-y",
            "-loglevel",
            self._get_ffmpeg_loglevel(),
        ]

        if input_path:
            cmd_list.extend(["-i", str(input_path)])

        cmd_list.extend(args)

        if output_path:
            cmd_list.append(str(output_path))

        self.logger.debug(f"Running FFmpeg command: {' '.join(cmd_list)}")

        try:
            start = time.perf_counter()

            process = subprocess.run(
                cmd_list, capture_output=True, text=True, timeout=self.timeout
            )

            if process.returncode != 0:
                self.logger.error(
                    f"FFmpeg command failed with return code {process.returncode}\n"
                    f"Error output: {process.stderr}"
                )
                raise subprocess.CalledProcessError(
                    process.returncode, cmd_list, process.stdout, process.stderr
                )

            end = time.perf_counter()
            self.logger.debug(f"FFmpeg command completed in {end - start:.4f} seconds")
            return process

        except subprocess.TimeoutExpired:
            self.logger.error("FFmpeg command timed out")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error running FFmpeg command: {str(e)}")
            raise

    def _get_frames_dir(self, *, sub_dir: str | None = None) -> Path:
        base = self.tmp_path / f"{self.input_path.stem}_{self.file_hash}"
        return base if not sub_dir else base / sub_dir

    def extract_frames(
        self,
        nth_frame: int = 120,
        output_format: str = "jpg",
    ) -> list[str]:
        """
        Extract frames from video at an interval. Store in temp directory.

        Args:
            input_path: Path to input video
            file_hash: Unique hash for the input video
            nth_frame: Extract every Nth frame (default: 1 = every frame)
            output_format: Output image format (default: jpg)

        Returns:
            Path to directory containing extracted frames
        """
        # Create unique output directory for frames
        output_dir = self._get_frames_dir(sub_dir="interval")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Construct output pattern for frames
        output_pattern = str(output_dir / f"interval_%d.{output_format}")

        self.logger.info(f"Extracting frames to {output_dir}.")

        args = [
            "-vf",
            f"select=not(mod(n\\,{nth_frame}))",
            "-frame_pts",
            "1",
            "-fps_mode",
            "vfr",
        ]

        self._run_command(args, self.input_path, output_pattern)

        # returns list of absolute file paths of extracted frames
        return get_filenames_of_dir(output_dir)

    def extract_frames_timestamps(
        self,
        timestamps: List[float],
        output_format: str = "jpg",
    ) -> list[Tuple[str, float]]:
        """
        Extract frames from video at specified timestamps

        Args:
            input_path: Path to input video
            file_hash: Unique hash for the input video
            timestamps: List of timestamps to extract frames at

        Returns:
            A list of (path, timestamp) tuples for extracted frames
        """
        # Create unique output directory for frames
        output_dir = self._get_frames_dir(sub_dir="timestamps")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Extracting frames at timestamp to {output_dir}.")

        for timestamp in tqdm(timestamps):
            # Create output filename
            output_pattern = os.path.join(
                output_dir,
                f"timestamp_{str(timestamp).replace('.', '_')}.{output_format}",
            )

            # Construct FFmpeg command
            args = [
                "-ss",
                str(timestamp),  # FFMPEG accepts a float as a timestamp
                "-i",
                str(self.input_path),
                "-vframes",
                "1",
                "-q:v",
                "2",
            ]

            self._run_command(args, self.input_path, output_pattern)

        paths = get_filenames_of_dir(output_dir)

        return list(zip(paths, timestamps))

    def extract_audio(
        self,
        output_format: str = "mp3",
        audio_bitrate: str = "192k",
        start_time: Optional[str] = None,
        duration: Optional[str] = None,
    ) -> Path:
        """
        Extract audio from video file

        Args:
            input_path: Path to input video
            output_format: Output audio format (default: mp3)
            audio_bitrate: Audio bitrate (default: 192k)
            start_time: Start time for extraction (optional, format: HH:MM:SS)
            duration: Duration to extract (optional, format: HH:MM:SS)

        Returns:
            Path to extracted audio file
        """
        output_path = (
            self.tmp_path / f"{self.input_path.stem}_{self.file_hash}.{output_format}"
        )
        self.logger.info(f"Extracting audio to {output_path}.")

        args = [
            "-vn",  # No video
            "-acodec",
            "libmp3lame" if output_format == "mp3" else "copy",
            "-ab",
            audio_bitrate,
        ]

        if start_time:
            args.extend(["-ss", start_time])
        if duration:
            args.extend(["-t", duration])

        self._run_command(args, self.input_path, output_path)
        return output_path
