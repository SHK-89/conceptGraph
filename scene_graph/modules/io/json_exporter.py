import json
import os
from scene_graph.modules.core.scene_graph import VideoSceneGraph


class SceneGraphJSONExporter:

    @staticmethod
    def save(scene_graph: VideoSceneGraph,
             output_dir: str):

        os.makedirs(output_dir, exist_ok=True)

        data = {
            "video": scene_graph.video_name,
            "frames": []
        }

        frame_ids = sorted(scene_graph.frame_objects.keys())

        for frame_idx in frame_ids:

            objects = scene_graph.get_objects_in_frame(frame_idx)

            edges = [
                e for e in scene_graph.edges
                if e.frame == frame_idx
            ]

            frame_entry = {
                "frame_index": frame_idx,
                "objects": [
                    {
                        "ojb_id": obj.obj_id,
                        "bbox": list(obj.bbox),
                        "center": list(obj.center)
                    }
                    for obj in objects
                ],
                "edges": [
                    {
                        "src": e.src_id,
                        "dst": e.dst_id,
                        "relation": e.relation
                    }
                    for e in edges
                ]
            }

            data["frames"].append(frame_entry)

        save_path = os.path.join(
            output_dir,
            f"{scene_graph.video_name}_scenegraph.json"
        )

        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

        return save_path
