
import os

from scenegraph.core.scene_graph import VideoSceneGraph
from scenegraph.io.json_exporter import SceneGraphJSONExporter
from scenegraph.utils.json_export import save_master_video_json
from scenegraph.visualization.scene_graph_visualizer import SceneGraphVisualizer
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

    scene_graph: VideoSceneGraph =  pipeline.run(
        video_path=video_path,
        video_name=video_name,
        annotation_pkl=ann_path,
        max_frames=config.max_frames
    )

    SceneGraphJSONExporter.save(scene_graph, output_dir)
    if config.draw_scene_graph:
        visualizer = SceneGraphVisualizer(output_dir=output_dir)
        visualizer.visualize_video(scene_graph, config.draw_overlay)

    print(f"✔ Finished {video_name}")

#TODO:SHK decide which version should I keep later.
    #if config.draw_scene_graph:
     #   visualize_all_frames(result)

    #save_master_video_json(result)
