import os
import json
import cv2
import numpy as np

from main import run_uvo_conceptgraphs_2d
from visualization.scenegraph2d_viz import visualize_scene_graph_2d


def find_uvo_pairs(root):
    """
    Finds UVO videos and their .pkl annotation pairs.
    """
    files = os.listdir(root)
    videos = [f for f in files if f.endswith(".mp4")]

    pairs = []
    for v in videos:
        base = v.replace(".mp4", "")
        pkl = base + ".pkl"
        if pkl in files:
            pairs.append((os.path.join(root, v), os.path.join(root, pkl)))

    return pairs


def batch_process_uvo_2d_intervl2(root, output_dir="uvo_scenegraphs_2d_intervl2",
                                  max_frames=40, save_visualization=True):
    os.makedirs(output_dir, exist_ok=True)

    pairs = find_uvo_pairs(root)
    print(f"Found {len(pairs)} UVO videos.")

    for video_path, pkl_path in pairs:
        name = os.path.basename(video_path).replace(".mp4", "")
        print(f"\n🔥 Processing {name} using InternVL2-26B-Chat …")

        scene_graph = run_uvo_conceptgraphs_2d_intervl2(
            video_path=video_path,
            annotation_pkl=pkl_path,
            max_frames=max_frames
        )

        # Save JSON scene graph
        json_path = os.path.join(output_dir, f"{name}_scenegraph.json")
        with open(json_path, "w") as f:
            json.dump(scene_graph, f, indent=4)

        print(f"✔ Saved scene graph → {json_path}")

        # Visualization
        if save_visualization:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                viz_path = os.path.join(output_dir, f"{name}_scenegraph.png")

                visualize_scene_graph_2d(
                    rgb,
                    scene_graph["objects"],
                    scene_graph["edges"],
                    display=False,
                    save_path=viz_path
                )

                print(f"✔ Saved visualization → {viz_path}")

            cap.release()

    print("\n🎉 Batch processing completed successfully!")
