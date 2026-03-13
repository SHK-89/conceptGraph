import cv2


class GazeObjectVisualizer:

    def draw_objects(self, frame, objects):

        for obj_id, obj in enumerate(objects):

            bbox = obj.get("box")

            if bbox is None or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (150,150,150),
                1
            )

        return frame


    def draw_gaze(self, frame, bbox):

        x1,y1,x2,y2 = map(int,bbox)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

        return frame


    def draw_match(self, frame, bbox):

        x1,y1,x2,y2 = map(int,bbox)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)

        return frame