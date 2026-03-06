import os
import cv2
import matplotlib.pyplot as plt
import networkx as nx

from scene_graph.modules.core.scene_graph import VideoSceneGraph
from scene_graph.modules.adapters.networkx_adapter import NetworkXAdapter


class SceneGraphVisualizer:

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------
    # Draw bounding boxes
    # -------------------------------------------------------
    @staticmethod
    def draw_objects_on_frame(rgb_frame, objects,
                              font_scale=0.6,
                              thickness=2):

        img = rgb_frame.copy()

        for obj in objects:
            x, y, w, h = map(int, obj.bbox)
            caption = obj.caption

            cv2.rectangle(img, (x, y), (x + w, y + h),
                          (0, 255, 0), thickness)

            cv2.rectangle(img,
                          (x, y - 20),
                          (x + len(caption) * 8, y),
                          (0, 255, 0), -1)

            cv2.putText(img,
                        caption,
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        thickness // 2,
                        cv2.LINE_AA)

        return img

    # -------------------------------------------------------
    # Draw relations on frame
    # -------------------------------------------------------
    @staticmethod
    def draw_relations_on_frame(rgb_frame, objects, edges):

        img = rgb_frame.copy()
        obj_map = {obj.obj_id: obj for obj in objects}

        for edge in edges:
            src = obj_map[edge.src_id]
            dst = obj_map[edge.dst_id]

            x1, y1 = map(int, src.center)
            x2, y2 = map(int, dst.center)

            cv2.arrowedLine(img, (x1, y1), (x2, y2),
                            (255, 0, 0), 2)

            mid = ((x1 + x2) // 2, (y1 + y2) // 2)

            cv2.putText(img,
                        edge.relation,
                        mid,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA)

        return img

    # -------------------------------------------------------
    # Draw NetworkX graph for one frame
    # -------------------------------------------------------
    @staticmethod
    def draw_scene_graph_graph(scene_graph: VideoSceneGraph,
                               frame_idx: int,
                               save_path=None):

        G = NetworkXAdapter.to_frame_graph(scene_graph, frame_idx)

        pos = nx.spring_layout(G, k=1.2, iterations=50)

        plt.figure(figsize=(12, 8))

        node_labels = {n: G.nodes[n]['caption'] for n in G.nodes()}
        edge_labels = {(u, v): d['relation']
                       for u, v, d in G.edges(data=True)}

        nx.draw_networkx_nodes(G, pos,
                               node_size=1500,
                               node_color="#85C1E9")

        nx.draw_networkx_edges(G, pos,
                               arrows=True,
                               arrowstyle='-|>',
                               arrowsize=20)

        nx.draw_networkx_labels(G, pos,
                                labels=node_labels,
                                font_size=9)

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_color="red",
            font_size=8
        )

        plt.title(f"Scene Graph — Frame {frame_idx}")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path,
                        dpi=200,
                        bbox_inches="tight")

        plt.close()

    # -------------------------------------------------------
    # Visualize all frames
    # -------------------------------------------------------
    def visualize_video(self, scene_graph: VideoSceneGraph, draw_overlay=True):

        for frame_idx, rgb in scene_graph.frames.items():

            objects = scene_graph.get_objects_in_frame(frame_idx)
            edges = [
                e for e in scene_graph.edges
                if e.frame == frame_idx
            ]

            # Draw image with objects
            vis_img = self.draw_objects_on_frame(rgb, objects)
            if draw_overlay:
                vis_img = self.draw_relations_on_frame(vis_img,
                                                   objects,
                                                   edges)

            if self.output_dir:
                cv2.imwrite(
                    os.path.join(
                        self.output_dir,
                        f"frame_{frame_idx}_overlay.jpg"
                    ),
                    cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                )

            # Draw graph
            if self.output_dir:
                graph_path = os.path.join(
                    self.output_dir,
                    f"frame_{frame_idx}_graph.jpg"
                )
            else:
                graph_path = None

            self.draw_scene_graph_graph(scene_graph,
                                        frame_idx,
                                        save_path=graph_path)
