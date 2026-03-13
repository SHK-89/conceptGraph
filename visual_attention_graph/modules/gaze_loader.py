import pandas as pd


class GazeLoader:

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load(self):

        df = pd.read_csv(self.csv_path)

        df = df[df["event"] == "FOV"]

        return df

    def filter_video(self, df, video_name):

        return df[df["filename"] == video_name]