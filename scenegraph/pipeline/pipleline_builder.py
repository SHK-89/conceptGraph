
from scenegraph.segmentation.uvo_segmenter import UVOSegmenter
from scenegraph.features.clip_encoder import CLIPEncoder
from scenegraph.captioning.interVLCaptioner import InterVLCaptioner
from scenegraph.pipeline.pipeline_runner import ConceptGraphsPipeline

from scenegraph.config.config import BatchConfig


def build_pipeline(config: BatchConfig) -> ConceptGraphsPipeline:
    """
    Initialize heavy models once.
    """

    segmenter = UVOSegmenter()
    clip_encoder = CLIPEncoder(device=config.device)

    #captioner = None
    #if config.enable_caption:
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
