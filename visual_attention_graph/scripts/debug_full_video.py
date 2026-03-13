import pandas as pd
import pickle

from visual_attention_graph.debug.gaze_object_visualizer import GazeObjectVisualizer
from visual_attention_graph.debug.video_debugger import VideoGazeDebugger
from visual_attention_graph.modules.bbox_mapper import BoundingBoxMapper


VIDEO_NAME = "3Qzzlg_SNGM.mp4"

VIDEO_PATH = rf"C:\SCIoI\Labrotation_SceneGraph\scene_graph\datasets\\uvoVideos\\{VIDEO_NAME}"
ANNOTATION_PATH = rf"C:\SCIoI\Labrotation_SceneGraph\scene_graph\datasets\\annotationfiles\\{VIDEO_NAME[:-4]}.pkl"
GAZE_CSV = rf"C:\SCIoI\Labrotation_SceneGraph\visual_attention_graph\data\eye_movement_data.csv"

OUTPUT_VIDEO = rf"C:\SCIoI\Labrotation_SceneGraph\\visual_attention_graph\scripts\outputs\{VIDEO_NAME}_debug_attention_mapping.mp4"


# -------------------------
# load gaze
# -------------------------

df = pd.read_csv(GAZE_CSV)

video_df = df[
    (df["filename"] == VIDEO_NAME) &
    (df["event"] == "FOV")
]


# -------------------------
# load annotations
# -------------------------

with open(ANNOTATION_PATH, "rb") as f:
    annotations = pickle.load(f)


# -------------------------
# initialize modules
# -------------------------

visualizer = GazeObjectVisualizer()

mapper = BoundingBoxMapper()

debugger = VideoGazeDebugger(
    visualizer,
    mapper
)


# -------------------------
# run debugger
# -------------------------

debugger.run_video(
    VIDEO_PATH,
    video_df,
    annotations,
    OUTPUT_VIDEO
)