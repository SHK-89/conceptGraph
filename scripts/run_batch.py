from scenegraph.pipeline.batch_runner import run_batch
from scenegraph.config.config import BatchConfig


if __name__ == "__main__":

    config = BatchConfig(
        video_dir="/home/shokoofeh/Labrotation_SceneGraph/datasets/uvoVideos",
        ann_dir="/home/shokoofeh/Labrotation_SceneGraph/datasets/annotationfiles",
        output_root="/home/shokoofeh/Labrotation_SceneGraph/experiments/results",

        max_frames=90,
        max_videos=20,
        device="cuda",

        draw_scene_graph=False,
        draw_overlay=False,
        store_crop=True,
        enable_caption=False,
        quiet_mode=True,
        resume=True
    )

    run_batch(config)
