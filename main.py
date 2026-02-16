import cv2
import torch

from captioning.interVLCaptioner import InterVLCaptioner
from features.clip_encoder import CLIPEncoder
from graph.graph_builder import build_2d_scene_graph_intervl2
from segmentation.uvo_segmenter import UVOSegmenter

from geometry.geom2d import bbox_center


def run_uvo_concept_graphs(video_path, annotation_pkl,
                             max_frames=2): #TODO SHK: change fram to 90 based on the dataset later
    """
    Pure 2D ConceptGraphs pipeline for UVO annotations using:
        - UVOSegmenter2D (PKL masks)
        - CLIP embeddings (2D)
        - InterVL3_5-8B (caption & relations)
        - 2D scene graph builder


    Returns:
        { "objects": [...], "edges": [...] }
    """

    # ---------------------------------------------------------
    # Load all modules
    # ---------------------------------------------------------
    print("Loading UVO PKL segmentations…")
    segmenter = UVOSegmenter(annotation_pkl)

    print("Initializing CLIP encoder…")
    clip_encoder = CLIPEncoder(device=torch.device('cuda'))

    print("Loading InterVL3_5-8B (captioner)…")
    captioner = InterVLCaptioner('OpenGVLab/InternVL3_5-8B',device=torch.device('cuda'))

    # ---------------------------------------------------------
    # Load video
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(video_path)

    # We store ALL object nodes from all frames
    objects = []

    print("Running ConceptGraphs-2D on video:", video_path)
    print("Max frames:", max_frames)
    counter =0
    # ---------------------------------------------------------
    # Main processing loop
    # ---------------------------------------------------------
    for frame_idx in segmenter.frame_indices:
        if counter >= max_frames:
            break
        counter +=1
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract segmentation for the frame
        masks, bboxes, instance_ids = segmenter.segment(frame_idx)

        print(f"Frame {frame_idx}: {len(masks)} objects")

        # -----------------------------------------------------
        # Process each object in the frame
        # -----------------------------------------------------
    for mask, bbox, oid in zip(masks, bboxes, instance_ids):
        # Masked crop for CLIP + captioning
        crop = rgb * mask[..., None]

        # 1) CLIP embedding
        feat = clip_encoder.encode_masked(rgb, mask)

        # 2) Object caption (InternVL3.5-8B-Chat)
        caption = captioner.caption_object(crop)
        print(f"  Object {oid} caption: {caption}")
        # 3) Compute center for 2D spatial relation
        center = bbox_center(bbox)

        # Store node
        objects.append({
            "frame": frame_idx,
            "id": oid,
            "bbox": bbox,
            "center": center,
            "feat": feat,
            "caption": caption
        })


    cap.release()

    print("Total objects collected:", len(objects))

    # ---------------------------------------------------------
    # Build final scene graph (2D)
    # ---------------------------------------------------------
    print("Building 2D scene graph using InterVL3_5-8B…")

    edges = build_2d_scene_graph_intervl2(objects, captioner)

    print("Scene graph built:")
    print("   • Nodes:", len(objects))
    print("   • Edges:", len(edges))

    # ---------------------------------------------------------
    # Return full scene graph
    # ---------------------------------------------------------
    return {
        "objects": objects,
        "edges": edges
    }


run_uvo_concept_graphs('/home/shokoofeh/Labrotation_SceneGraph/datasets/uvoVideos/--8FQVwWH0M.mp4',
                         '/home/shokoofeh/Labrotation_SceneGraph/datasets/annotationfiles/--8FQVwWH0M.pkl')
