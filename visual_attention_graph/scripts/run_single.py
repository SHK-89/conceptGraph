from visual_attention_graph.modules.pipeline.attention_pipeline import AttentionPipeline
from visual_attention_graph.modules.graph_exporter import GraphExporter


GAZE_CSV = r"C:\SCIoI\Labrotation_SceneGraph\visual_attention_graph\data\eye_movement_data.csv"

ANNOTATION_DIR = r"C:\SCIoI\Labrotation_SceneGraph\scene_graph\datasets\annotationfiles"

VIDEO_DIR = r"C:\SCIoI\Labrotation_SceneGraph\scene_graph\datasets\uvoVideos"

VIDEO_NAME = "-0CNBzthkZ4.mp4"


pipeline = AttentionPipeline(
    GAZE_CSV,
    ANNOTATION_DIR
)

graph = pipeline.run_single_video(VIDEO_NAME)


exporter = GraphExporter(output_dir="outputs")

exporter.save(graph, VIDEO_NAME)

print("Nodes:", graph.number_of_nodes())
print("Edges:", graph.number_of_edges())