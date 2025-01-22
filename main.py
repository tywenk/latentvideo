import argparse
import time
import warnings

from tqdm import tqdm
from textwrap import shorten

from collection import Collection
from ffmpeg import FFmpeg
from filehash import get_file_hash
from florence import Florence
from logger import Logger
from whisper import Whisper

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = Logger.setup(__name__)


def main():
    parser = argparse.ArgumentParser(description="Embed a video file")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()

    video_path = str(args.video_path)

    collection = Collection()
    whisper = Whisper(input_path=video_path)
    florence = Florence(input_path=video_path)
    ffmpeg = FFmpeg(
        input_path=video_path,
        ffmpeg_path="ffmpeg",
        tmp_path="./tmp/",
        timeout=3600,
    )

    # STEP 1: Extract audio from video and transcribe
    audio_chunks = []
    try:
        audio_path = ffmpeg.extract_audio()
        audio_chunks = whisper.transcribe_audio(audio_path)

        logger.info(f"Transcribed {len(audio_chunks)} audio chunks")

        collection.add_text(
            ids=[
                f"a_{i}_{shorten(get_file_hash((video_path)), width=59)}"
                for i in range(len(audio_chunks))
            ],
            documents=[ac["text"] for ac in audio_chunks],
            metadatas=[{"timestamp": str(ac["timestamp"])} for ac in audio_chunks],
        )
    except Exception as e:
        logger.error(f"Step 1 Error: {str(e)}")
        raise

    # STEP 2: Extract frames from video at set intervals
    intv_frames_paths = []
    try:
        intv_frames_paths = ffmpeg.extract_frames(nth_frame=120)
    except Exception as e:
        logger.error(f"Step 2 Error: {str(e)}")
        raise

    # STEP 3: Extract frames from video at set intervals
    ts_frames_paths = []
    try:
        # Get the starting timestamp of each audio chunk
        timestamps = [float(ac["timestamp"][0]) for ac in audio_chunks]  # type: ignore
        ts_frames_paths = ffmpeg.extract_frames_timestamps(timestamps=timestamps)
    except Exception as e:
        logger.error(f"Step 3 Error: {str(e)}")
        raise

    # STEP 4: Extract caption of timestamped frames
    caption_texts = []
    try:
        logger.info("Generating captions for timestamped frames")
        start = time.perf_counter()
        for path, _ in tqdm(ts_frames_paths):
            caption = florence.get_video_caption(image_path=path, prompt="<CAPTION>")
            caption_texts.append((caption, path))
        end = time.perf_counter()
        logger.info(
            f"Caption generation completed for {len(ts_frames_paths)} frames in {end - start:.4f} seconds"
        )

        logger.info("Generating captions for interval frames")
        start = time.perf_counter()
        for path in tqdm(intv_frames_paths):
            caption = florence.get_video_caption(image_path=path, prompt="<CAPTION>")
            caption_texts.append((caption, path))
        end = time.perf_counter()
        logger.info(
            f"Caption generation completed for {len(ts_frames_paths)} frames in {end - start:.4f} seconds"
        )
    except Exception as e:
        logger.error(f"Step 4 Error: {str(e)}")
        raise

    logger.info("All steps completed successfully")


if __name__ == "__main__":
    main()
