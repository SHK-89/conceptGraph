from pipeline.pipeline_runner import ConceptGraphsPipeline
from utils.vis import visualize_all_frames, draw_objects_on_frame, draw_scene_graph, visualize_full_pipeline

video = '/home/shokoofeh/Labrotation_SceneGraph/datasets/uvoVideos/--8FQVwWH0M.mp4'
pkl = '/home/shokoofeh/Labrotation_SceneGraph/datasets/annotationfiles/--8FQVwWH0M.pkl'


print("Running ConceptGraphs pipeline...")
pipeline = ConceptGraphsPipeline(device="cuda")
result = pipeline.run(video, pkl, max_frames=2)

print("Graph nodes:", len(result["objects"]))
print("Graph edges:", len(result["edges"]))

objects = result["objects"]
edges = result["edges"]

draw_scene_graph(objects, edges, save_path="/home/shokoofeh/Labrotation_SceneGraph/results/output/scene_graph.png")

# visualize_all_frames(result["frames"], result["frame_objects"], result["edges"],
#                      out_dir="/home/shokoofeh/Labrotation_SceneGraph/results/output/frames/")
# visualize_full_pipeline(first_rgb_frame, objects, edges)
