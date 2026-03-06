from scene_graph.modules.segmentation.uvo_segmenter import UVOSegmenter


def load_uvo_video_dataset(video_path, annotation_path, K, max_frames=None):
    import cv2
    cap = cv2.VideoCapture(video_path)
    segmenter = UVOSegmenter(annotation_path)

    rgb_frames = []
    depth_frames = []

    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth = predict_depth(frame_rgb)

        rgb_frames.append(frame_rgb)
        depth_frames.append(depth)

        if max_frames and idx >= max_frames:
            break
        idx += 1

    poses = estimate_poses(rgb_frames, depth_frames, K)

    dataset = []
    for i, (rgb, depth, pose) in enumerate(zip(rgb_frames, depth_frames, poses)):
        masks, instance_ids = segmenter.segment(i + 1, rgb)
        dataset.append({
            "rgb": rgb,
            "depth": depth,
            "pose": pose,
            "masks": masks,
            "ids": instance_ids
        })

    return dataset