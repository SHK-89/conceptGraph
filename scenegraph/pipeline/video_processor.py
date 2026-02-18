
import os

from scenegraph.utils.json_export import save_master_video_json
from scenegraph.visualization.vis import visualize_all_frames

from scenegraph.config.config import BatchConfig


def process_single_video(
        pipeline,
        video_name: str,
        video_path: str,
        ann_path: str,
        config: BatchConfig
):
    """
    Process one video safely.
    """

    output_dir = os.path.join(config.output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    master_json_path = os.path.join(output_dir, "master.json")

    # Resume protection
    if config.resume and os.path.exists(master_json_path):
        print(f"Skipping {video_name} (already processed).")
        return

    result = pipeline.run(
        video_path=video_path,
        video_name=video_name,
        annotation_pkl=ann_path,
        max_frames=config.max_frames
    )

    result["output_dir"] = output_dir
    result["video_name"] = video_name

    if config.draw_scene_graph:
        visualize_all_frames(result)

    save_master_video_json(result)
