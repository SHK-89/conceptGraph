import cv2
from tqdm import tqdm
from typing import Tuple, Dict, List
from scene_graph.modules.core.models import DetectedObject
from scene_graph.modules.geometry.geom2d import bbox_center


class ObjectExtractionStage:

    def __init__(self, segmenter, clip_encoder, captioner=None, enable_caption=False, store_crop=False):
        self.segmenter = segmenter
        self.clip = clip_encoder
        self.captioner = captioner
        self.enable_caption = enable_caption
        self.store_crop= store_crop

    def run(
        self,
        video_path: str,
        video_name: str,
        max_frames: int
    ) -> Tuple[List[DetectedObject], Dict[int, List[DetectedObject]], Dict[int, any]]:

        cap = cv2.VideoCapture(video_path)

        objects = []
        frame_objects = {}
        frames_rgb = {}

        for frame_idx in tqdm(
                self.segmenter.frame_indices[:max_frames],
                desc=f"Frames ({video_name})",
                leave=False):

            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_rgb[frame_idx] = rgb.copy()
            frame_objects[frame_idx] = []

            masks, bboxes, ids = self.segmenter.segment(frame_idx)

            for mask, bbox, obj_id in zip(masks, bboxes, ids):

                crop = rgb * mask[..., None]
                feat = self.clip.encode_masked(crop, mask)

                caption = "unknown"
                if self.enable_caption and self.captioner:
                    cap_out = self.captioner.caption_object(crop)
                    if cap_out:
                        caption = cap_out

                center = bbox_center(bbox)

                obj = DetectedObject(
                    frame=frame_idx,
                    obj_id=obj_id,
                    bbox=bbox,
                    center=center,
                    caption=caption,
                    feature=feat,
                    crop=crop if self.store_crop else None  # avoid memory explosion
                )

                objects.append(obj)
                frame_objects[frame_idx].append(obj)

        cap.release()
        return objects, frame_objects, frames_rgb
