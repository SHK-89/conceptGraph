import pickle
import os


class AnnotationLoader:

    def __init__(self, annotation_dir):

        self.annotation_dir = annotation_dir
        self.cache = {}

    def load(self, video_name):

        if video_name in self.cache:
            return self.cache[video_name]

        name = os.path.splitext(video_name)[0]

        path = os.path.join(self.annotation_dir, f"{name}.pkl")

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.cache[video_name] = data

        return data