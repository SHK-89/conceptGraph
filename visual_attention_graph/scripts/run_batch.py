from modules.pipeline import VisualAttentionGraphPipeline
from modules.visualization import GraphVisualizer

CSV_PATH = "data/eye_movement_data.csv"

pipeline = VisualAttentionGraphPipeline(CSV_PATH)

graphs = pipeline.run()

print("Graphs created:", len(graphs))