import argparse
import time
from typing import Tuple
import warnings
from pprint import pp

from tqdm import tqdm
from textwrap import shorten

from collection import Collection, CollectionId
from ffmpeg import FFmpeg
from filehash import get_file_hash
from florence import Florence
from logger import Logger
from whisper import Whisper

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = Logger.setup(__name__)


def extract_audio(ffmpeg: FFmpeg, whisper: Whisper, collection: Collection):
    """Extract audio from video and transcribe"""
    try:
        audio_path = ffmpeg.extract_audio()
        audio_chunks = whisper.transcribe_audio(audio_path)
        collection.add_text(
            kind=CollectionId.AUDIO_EXTRACT,
            documents=[ac["text"] for ac in audio_chunks],
            metadatas=[{"timestamp": str(ac["timestamp"])} for ac in audio_chunks],
        )
        return audio_chunks
    except Exception as e:
        logger.error(f"Step 1 Error: {str(e)}")
        raise


def extract_frames(
    ffmpeg: FFmpeg,
    collection: Collection,
):
    """Extract frames from video at set intervals"""
    try:
        intv_frames_paths = ffmpeg.extract_frames(nth_frame=120)
        collection.add_image(
            kind=CollectionId.FRAME_EXTRACT,
            uris=[str(path) for path, _ in intv_frames_paths],
            metadatas=[{"timestamp": str(ts)} for _, ts in intv_frames_paths],
        )
        return intv_frames_paths
    except Exception as e:
        logger.error(f"Step 2 Error: {str(e)}")
        raise


def extract_frames_intv(
    ffmpeg: FFmpeg, collection: Collection, timestamps: list[float]
):
    """Extract frames from video at set intervals"""
    try:
        # Get the starting timestamp of each audio chunk
        # timestamps = [float(ac["timestamp"][0]) for ac in audio_chunks]  # type: ignore
        return ffmpeg.extract_frames_timestamps(timestamps=timestamps)
    except Exception as e:
        logger.error(f"Step 3 Error: {str(e)}")
        raise


def caption_frames(
    florence: Florence, collection: Collection, ts_frames_paths: list[Tuple[str, float]]
):
    """Get captions for frames"""
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
    except Exception as e:
        logger.error(f"Step 4 Error: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Embed a video file")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    args = parser.parse_args()

    video_path = str(args.video_path)

    collection = Collection(input_path=video_path)
    whisper = Whisper(input_path=video_path)
    florence = Florence(input_path=video_path)
    ffmpeg = FFmpeg(
        input_path=video_path,
        ffmpeg_path="ffmpeg",
        tmp_path="./tmp/",
        timeout=3600,
    )

    extract_audio(ffmpeg, whisper, collection)
    frames = extract_frames(ffmpeg, collection)
    caption_frames(florence, collection, frames)

    logger.info("All steps completed successfully")

    query = input("Enter a query: ")

    img_res, text_res = collection.search(query=query)

    # TODO: Add better printing of results
    pp(img_res)
    pp(text_res)


if __name__ == "__main__":
    main()
