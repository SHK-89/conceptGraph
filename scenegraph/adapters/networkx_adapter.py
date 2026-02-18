import networkx as nx
from scenegraph.core.scene_graph import VideoSceneGraph


class NetworkXAdapter:

    @staticmethod
    def to_frame_graph(scene_graph: VideoSceneGraph, frame_idx: int):
        """
        Create a graph for a single frame.
        Nodes: objects
        Edges: relations
        """

        G = nx.DiGraph()

        # Add nodes
        for obj in scene_graph.get_objects_in_frame(frame_idx):
            G.add_node(
                obj.obj_id,
                caption=obj.caption,
                bbox=obj.bbox,
                center=obj.center
            )

        # Add edges
        for edge in scene_graph.edges:
            if edge.frame == frame_idx:
                G.add_edge(
                    edge.src_id,
                    edge.dst_id,
                    relation=edge.relation
                )

        return G

    @staticmethod
    def to_video_graph(scene_graph: VideoSceneGraph):
        """
        Create full multi-frame graph.
        Nodes uniquely identified by (frame, obj_id).
        """

        G = nx.DiGraph()

        # Add nodes
        for obj in scene_graph.objects:
            node_id = (obj.frame, obj.obj_id)

            G.add_node(
                node_id,
                frame=obj.frame,
                caption=obj.caption,
                bbox=obj.bbox,
                center=obj.center
            )

        # Add edges
        for edge in scene_graph.edges:
            src = (edge.frame, edge.src_id)
            dst = (edge.frame, edge.dst_id)

            G.add_edge(
                src,
                dst,
                relation=edge.relation
            )

        return G
