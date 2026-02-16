import cv2
from captioning.interVLCaptioner import InterVLCaptioner
from features.clip_encoder import CLIPEncoder
from geometry.geom2d import bbox_center, union_bbox, crop_pair_region, remap_center
from graph.graph_builder import build_2d_scene_graph_intervl
from segmentation.uvo_segmenter import UVOSegmenter
from tqdm import tqdm


class ConceptGraphsPipeline:
    """Full UVO → Object Embedding → Caption → Relations → Scene Graph pipeline."""

    def __init__(self, segmenter, clip_encoder, captioner, device="cuda", enable_caption=False, quiet=False):
        print("Initializing pipeline modules...")
        self.device = device
        self.segmenter = segmenter
        self.clip = clip_encoder
        self.captioner = captioner
        self.quiet = quiet
        self.enable_caption = enable_caption  # Set to True to enable captioning (or False to testing or if captioner is unavailable)

    def log(self, **args):
        if not self.quiet:
            print(**args)

    def run_OLD(self, video_path, annotation_pkl, max_frames=2):  # todo: change max frams later
        self.load_modules(annotation_pkl)

        cap = cv2.VideoCapture(video_path)
        objects = []
        frame_objects = {}
        frames_rgb = {}

        fcount = 0

        print("Processing video…")

        for frame_idx in self.segmenter.frame_indices:
            if fcount >= max_frames:
                break
            fcount += 1

            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_rgb[frame_idx] = rgb.copy()
            frame_objects[frame_idx] = []

            masks, bboxes, ids = self.segmenter.segment(frame_idx)

            print(f"[Frame {frame_idx}]  {len(masks)} objects")

            for mask, bbox, obj_id in zip(masks, bboxes, ids):
                crop = rgb * mask[..., None]
                feat = self.clip.encode_masked(crop, mask)
                caption = self.captioner.caption_object(crop)
                center = bbox_center(bbox)
                obj = {
                    "frame": frame_idx,
                    "id": obj_id,
                    "bbox": bbox,
                    "center": center,
                    "caption": (caption if caption != null and enable_caption is True else "unknown"),
                    "feat": feat
                }
                objects.append(obj)
                frame_objects[frame_idx].append(obj)

        cap.release()

        print("Building scene graph…")
        edges = build_2d_scene_graph_intervl(objects, self.captioner)

        return {"objects": objects,
                "edges": edges,
                "frame_objects": frame_objects,
                "frames": frames_rgb
                }

    def extract_object(self, video_path, video_name, max_frames):
        print("Extracting objects...")
        cap = cv2.VideoCapture(video_path)
        objects = []
        frame_objects = {}
        frames_rgb = {}

        fcount = 0
        for frame_idx in tqdm(self.segmenter.frame_indices[:max_frames],
                              desc=f"Frames ({video_name})",
                              leave=False):
            if fcount >= max_frames:
                break
            fcount += 1

            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_rgb[frame_idx] = rgb.copy()
            frame_objects[frame_idx] = []

            masks, bboxes, ids = self.segmenter.segment(frame_idx)

            print(f"[Frame {frame_idx}]  {len(masks)} objects")
            for mask, bbox, obj_id in tqdm(zip(masks, bboxes, ids),
                                           total=len(masks),
                                           desc=f"Objs F{frame_idx}",
                                           leave=False):

                crop = rgb * mask[..., None]

                feat = self.clip.encode_masked(crop, mask)
                caption = None
                if self.enable_caption:
                    caption = self.captioner.caption_object(crop)
                center = bbox_center(bbox)

                obj = {
                    "frame": frame_idx,
                    "id": obj_id,
                    "bbox": bbox,
                    "center": center,
                    "caption": (caption if self.enable_caption is True and caption is not None and caption != "" else "unknown"),
                    "feat": feat,
                    "crop": crop
                }
                objects.append(obj)
                frame_objects[frame_idx].append(obj)

        cap.release()
        return objects, frame_objects, frames_rgb

    # ------------------------------------------------------------
    # RELATIONSHIP REASONING: pairwise InterVL visual reasoning
    # ------------------------------------------------------------
    def infer_relation(self, frame_objects, frames_rgb):
        print("Infer relation via InterVL inference reasoning…")

        edges = []
        for frame_idx, objs in frame_objects.items():
            rgb = frames_rgb[frame_idx]
            n = len(objs)
            print(f"Processing relationships for frame {frame_idx} with {n} objects.")
            for i in range(n):
                for j in range(i + 1, n):
                    objA = objs[i]
                    objB = objs[j]

                    # 1 Crop region containing both objects
                    region, offset = crop_pair_region(
                        rgb,
                        objA["bbox"],
                        objB["bbox"]
                    )

                    # 2 Compute new positions relative to the crop
                    cA = remap_center(objA["center"], offset)
                    cB = remap_center(objB["center"], offset)

                    # 3 Ask InterVL directly
                    relation = self.captioner.relation_from_paircrop(
                        region, cA, cB
                    )

                    # Debug print, remove or comment out in production
                    # print(f"Relation between object: {objA['caption']} and object: {objB['caption']} is:**** {relation}****")

                    edges.append({
                        "frame": frame_idx,
                        "src": objA["id"],
                        "dst": objB["id"],
                        "relation": relation
                    })

        print("Scene graph complete.")
        return edges

    def run(self, video_path, video_name, annotation_pkl, max_frames=90):

        self.segmenter.load(annotation_pkl)

        objects, frame_objects, frames_rgb = self.extract_object(video_path,
                                                                 video_name,
                                                                 max_frames=max_frames)
        edges = self.infer_relation(frame_objects, frames_rgb)

        return {"objects": objects,
                "edges": edges,
                "frame_objects": frame_objects,
                "frames": frames_rgb
                }
