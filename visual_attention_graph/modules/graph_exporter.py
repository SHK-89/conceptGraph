import json
import os
import networkx as nx
import matplotlib.pyplot as plt


class GraphExporter:

    def __init__(self, output_dir="outputs"):

        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)


    def save(self, graph, video_name):

        video_prefix = os.path.splitext(video_name)[0]
        video_dir = os.path.join(self.output_dir, video_prefix)
        os.makedirs(video_dir, exist_ok=True)

        json_path = os.path.join(video_dir, "attention_graph.json")
        img_path = os.path.join(video_dir, "attention_graph.png")

        self._save_json(graph, json_path)
        self._save_image(graph, img_path)

        print("Saved JSON:", json_path)
        print("Saved Image:", img_path)


    def _save_json(self, graph, path):

        data = nx.node_link_data(graph)

        with open(path, "w") as f:
            json.dump(data, f, indent=4)


    def _save_image(self, graph, path):

        plt.figure(figsize=(8,6))

        pos = nx.spring_layout(graph)

        weights = nx.get_edge_attributes(graph, "weight")

        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=2000,
            font_size=10
        )

        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=weights
        )

        plt.title("Visual Attention Graph")

        plt.savefig(path)

        plt.close()