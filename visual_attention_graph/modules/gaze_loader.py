import pandas as pd


class GazeDataLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load(self):
        df = pd.read_csv(self.csv_path)

        # keep only foveations
        df = df[df["event"] == "FOV"]

        return df