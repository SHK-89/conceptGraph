from typing import List, Dict
from scene_graph.modules.core.models import RelationEdge
from scene_graph.modules.geometry.geom2d import crop_pair_region, remap_center


class RelationInferenceStage:

    def __init__(self, captioner):
        self.captioner = captioner

    def run(
        self,
        frame_objects: Dict[int, List],
        frames_rgb: Dict[int, any]
    ) -> List[RelationEdge]:

        edges = []

        for frame_idx, objs in frame_objects.items():
            rgb = frames_rgb[frame_idx]
            n = len(objs)

            for i in range(n):
                for j in range(i + 1, n):
                    objA = objs[i]
                    objB = objs[j]

                    region, offset = crop_pair_region(
                        rgb,
                        objA.bbox,
                        objB.bbox
                    )

                    cA = remap_center(objA.center, offset)
                    cB = remap_center(objB.center, offset)

                    relation = self.captioner.relation_from_paircrop(
                        region, cA, cB
                    )

                    edges.append(
                        RelationEdge(
                            frame=frame_idx,
                            src_id=objA.obj_id,
                            dst_id=objB.obj_id,
                            relation=relation
                        )
                    )

        return edges
