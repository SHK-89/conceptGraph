from annotation.frame_object_analyzer import FrameObjectAnalyzer


ANNOTATION_DIR = "C:\SCIoI\Labrotation_SceneGraph\scene_graph\datasets\\annotationfiles"


analyzer = FrameObjectAnalyzer(ANNOTATION_DIR)


# --------------------------------------------------
# Step 1: count objects per frame
# --------------------------------------------------

frame_df = analyzer.analyze_all_frames()

print("\nObjects per frame:\n")
print(frame_df.head(20))


# --------------------------------------------------
# Step 2: group by video and compute statistics
# --------------------------------------------------

video_df = analyzer.summarize_by_video(frame_df)

print("\nVideo statistics (sorted by complexity):\n")
print(video_df)


# --------------------------------------------------
# Step 3: save results
# --------------------------------------------------

analyzer.save_results(frame_df, video_df)