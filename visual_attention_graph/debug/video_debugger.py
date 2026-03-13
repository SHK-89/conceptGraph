import cv2
import ast


class VideoGazeDebugger:

    def __init__(self, visualizer, mapper):

        self.visualizer = visualizer
        self.mapper = mapper


    def run_video(
        self,
        video_path,
        gaze_df,
        annotations,
        output_path
    ):

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_id = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            objects = annotations.get(str(frame_id), [])

            frame = self.visualizer.draw_objects(frame, objects)

            # find gaze rows covering this frame
            rows = gaze_df[
                (gaze_df["frame_start"] <= frame_id) &
                (gaze_df["frame_end"] >= frame_id)
            ]

            for _, row in rows.iterrows():

                bbox_dict = ast.literal_eval(
                    row["gt_object_multiple_bbox"]
                )

                if str(frame_id) not in bbox_dict:
                    continue

                for gaze_bbox in bbox_dict[str(frame_id)]:

                    frame = self.visualizer.draw_gaze(frame, gaze_bbox)

                    obj_id = self.mapper.map_bbox(
                        frame_id,
                        gaze_bbox,
                        annotations
                    )

                    if obj_id is not None:

                        obj_bbox = objects[obj_id]["box"]

                        frame = self.visualizer.draw_match(
                            frame,
                            obj_bbox
                        )

            writer.write(frame)

            frame_id += 1

        cap.release()
        writer.release()

        print("Saved debug video:", output_path)