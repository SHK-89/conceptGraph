import pickle
import numpy as np
import pycocotools.mask as mask_util


class UVOSegmenter:
    """
    Segmentation loader for UVO PKL annotations.

    Each frame key contains a list of objects:
        {
          'box': [x, y, w, h],
          'seg': RLE segmentation,
          ...
        }
    """

    def __init__(self):
        self.data = None
        self.frame_indices = []

    def load(self, annotation_pkl_path):
        with open(annotation_pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.frame_indices = sorted(map(int, self.data.keys()))


    def segment(self, frame_idx):
        print("Segmenting frame %d" % frame_idx)
        """
        Returns for a given frame:
            masks  : list of bool arrays (HxW)
            bboxes : list of [x, y, w, h]
            ids    : list of integer instance IDs (0..N-1)
        """
        key = str(frame_idx)
        if key not in self.data:
            return [], [], []

        objs = self.data[key]

        masks = []
        bboxes = []
        ids = []

        for inst_id, ann in enumerate(objs):
            # ----------------------------------------
            # Skip invalid objects
            # ----------------------------------------
            if ann["box"] is None:
                continue
            if ann["seg"] is None:
                continue
            if ann["seg"]["counts"] is None:
                continue

            box = ann["box"]
            if box == [0.0, 0.0, 0.0, 0.0]:
                continue  # empty box → skip

            # ----------------------------------------
            # Decode mask
            # ----------------------------------------
            try:
                mask = mask_util.decode(ann["seg"]).astype(bool)
            except Exception as e:
                print(f"[WARNING] Could not decode RLE for frame {frame_idx}: {e}")
                continue

            masks.append(mask)
            bboxes.append(box)
            ids.append(inst_id)  # always a clean integer ID

        return masks, bboxes, ids
