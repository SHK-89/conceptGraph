import json
import os

def safe_list(x):
    if isinstance(x, (list, tuple)):
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def save_master_video_json(result):
    """
    Save a single JSON file containing:
        - all frames
        - all objects per frame
        - all relations per frame
    param results:
    contains:
        video_name: str
        frame_objects: dict frame_idx → list of objects
        edges: list of dicts {frame, src, dst, relation}
        output_dir: folder to save the JSON file
    """
    video_name = result["video_name"]
    frame_objects = result["frame_objects"]
    edges = result["edges"]
    output_dir = result["output_dir"]

    data = {
        "video": video_name,
        "frames": []
    }

    # Collect unique frames
    frame_ids = sorted(frame_objects.keys())

    for frame_idx in frame_ids:
        objs = frame_objects[frame_idx]

        # Filter edges belonging to this frame
        frame_edges = [
            e for e in edges if e["frame"] == frame_idx
        ]

        frame_entry = {
            "frame_index": frame_idx,
            "objects": [
                {
                    "id": obj["id"],
                    "bbox": safe_list(obj["bbox"]),
                    "center": safe_list(obj["center"]),
                }
                for obj in objs
            ],
            "edges": [
                {
                    "src": e["src"],
                    "dst": e["dst"],
                    "relation": e["relation"]
                }
                for e in frame_edges
            ]
        }

        data["frames"].append(frame_entry)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{video_name}_scenegraph.json")

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Master JSON saved: {save_path}")
