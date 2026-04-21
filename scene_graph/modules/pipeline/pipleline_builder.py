
from scene_graph.modules.segmentation.uvo_segmenter import UVOSegmenter
from scene_graph.modules.features.clip_encoder import CLIPEncoder
from scene_graph.modules.captioning.interVLCaptioner import InterVLCaptioner
from scene_graph.modules.pipeline.concept_graph_pipeline import ConceptGraphsPipeline

from scene_graph.modules.config.config import BatchConfig


def build_pipeline(config: BatchConfig) -> ConceptGraphsPipeline:
    """
    Initialize heavy models once.
    """

    segmenter = UVOSegmenter()
    clip_encoder = CLIPEncoder(device=config.device)

    captioner = InterVLCaptioner(
            device=config.device,
            quiet_mode=config.quiet_mode
        )

    pipeline = ConceptGraphsPipeline(
        segmenter=segmenter,
        clip_encoder=clip_encoder,
        captioner=captioner,
        device=config.device,
        enable_caption=config.enable_caption
    )

    return pipeline
