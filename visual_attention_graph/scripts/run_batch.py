from visual_attention_graph.modules.graph_exporter import GraphExporter
from visual_attention_graph.modules.pipeline.attention_pipeline import AttentionPipeline


GAZE_CSV = "C:\SCIoI\Labrotation_SceneGraph\\visual_attention_graph\data\eye_movement_data.csv"
ANNOTATION_DIR = "C:\SCIoI\Labrotation_SceneGraph\scene_graph\datasets\\annotationfiles"


pipeline = AttentionPipeline(
    GAZE_CSV,
    ANNOTATION_DIR
)

exporter = GraphExporter(output_dir="outputs")


graphs = pipeline.run_all_videos()


for video, graph in graphs.items():

    exporter.save(graph, video)

    print(
        video,
        "nodes:", graph.number_of_nodes(),
        "edges:", graph.number_of_edges()
    )