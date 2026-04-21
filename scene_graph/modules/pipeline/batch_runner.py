import time
import traceback
from tqdm import tqdm

from scene_graph.modules.data.uvo_dataset import UVOVideoDataset
from scene_graph.modules.pipeline.pipleline_builder import build_pipeline
from scene_graph.modules.pipeline.video_processor import process_single_video
from scene_graph.modules.utils.loggin_utils import setup_logger
from scene_graph.modules.utils.progress import update_progress_bar
from scene_graph.modules.config.config import BatchConfig


def run_batch(config: BatchConfig):

    logger = setup_logger()

    logger.info("Starting batch processing...")

    dataset = UVOVideoDataset(
        video_dir=config.video_dir,
        ann_dir=config.ann_dir
    )

    pipeline = build_pipeline(config)

    start_time = time.time()
    video_count = 0

    pbar = tqdm(dataset, desc="Processing videos", dynamic_ncols=True)

    for video_name, video_path, ann_path in pbar:

        if config.max_videos and video_count >= config.max_videos:
            logger.info("Reached max video limit. Stopping.")
            print("Reached max video limit. Stopping.")
            break

        video_count += 1

        update_progress_bar(pbar, start_time)

        try:
            process_single_video(
                pipeline=pipeline,
                video_name=video_name,
                video_path=video_path,
                ann_path=ann_path,
                config=config
            )

        except Exception as e:
            logger.error(f"Error in {video_name}: {e}")
            traceback.print_exc()
            continue

    logger.info("Batch processing complete.")
