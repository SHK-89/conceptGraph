from scene_graph.modules.core.scene_graph import VideoSceneGraph
from scene_graph.modules.stages.object_extraction import ObjectExtractionStage
from scene_graph.modules.stages.relation_inference import RelationInferenceStage


class ConceptGraphsPipeline:

    def __init__(self, segmenter, clip_encoder, captioner=None,
                 device="cuda", enable_caption=False, store_crop=True):

        self.segmenter = segmenter
        self.clip = clip_encoder
        self.captioner = captioner
        self.device = device
        self.store_crop = store_crop

        self.object_stage = ObjectExtractionStage(
            segmenter=segmenter,
            clip_encoder=clip_encoder,
            captioner=captioner,
            enable_caption=enable_caption,
            store_crop=store_crop
        )

        self.relation_stage = RelationInferenceStage(
            captioner=captioner
        )

    def run(self, video_path, video_name, annotation_pkl, max_frames=90):

        # Load segmentation annotations
        self.segmenter.load(annotation_pkl)

        # 1️⃣ Extract objects
        objects, frame_objects, frames_rgb = self.object_stage.run(
            video_path,
            video_name,
            max_frames
        )

        # 2️⃣ Infer relations
        edges = self.relation_stage.run(
            frame_objects,
            frames_rgb
        )

        # 3️⃣ Assemble final graph
        scene_graph = VideoSceneGraph(
            video_name=video_name,
            objects=objects,
            edges=edges,
            frame_objects=frame_objects,
            frames=frames_rgb
        )

        return scene_graph
