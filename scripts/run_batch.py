from scenegraph.pipeline.batch_runner import run_batch
from scenegraph.config.config import BatchConfig


if __name__ == "__main__":

    config = BatchConfig(
        video_dir="/home/shokoofeh/Labrotation_SceneGraph/datasets/uvoVideos",
        ann_dir="/home/shokoofeh/Labrotation_SceneGraph/datasets/annotationfiles",
        output_root="/home/shokoofeh/Labrotation_SceneGraph/experiments/results",

        max_frames=3,
        max_videos=2,
        device="cuda",

        draw_scene_graph=True,
        enable_caption=False,
        quiet_mode=True,
        resume=True
    )

    run_batch(config)
