import os

import cv2
import matplotlib.pyplot as plt
import networkx as nx


def draw_scene_graph_OLD(objects, edges, save_path=None, show=True):
    """
    Visualize graph with:
      - Node: object caption or ID
      - Edge: relationship label
    """
    G = nx.DiGraph()


    # Add nodes
    for obj in objects:
        #G.add_node(obj["id"], label=obj["caption"])
        G.add_node(obj["id"], label=obj["id"]) # remove caption for now, just show ID to avoid clutter.

    # src dst
    # Add edges
    for edge in edges:
        G.add_edge(edge["src"], edge["dst"], label=edge["relation"])

    pos = nx.spring_layout(G, seed=42)

    labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, labels=labels,
            node_color='lightblue', node_size=1500,
            font_size=10, font_weight='bold', arrowsize=20)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def draw_scene_graph(objects, edges, save_path=None):
    """
    Visualize the scene graph using NetworkX.
    Nodes = object IDs
    Node labels = captions
    Edge labels = relationship (InterVL predicted)
    """
    G = nx.DiGraph()

    # Add nodes
    for obj in objects:
        # G.add_node(obj["id"], label=obj["caption"])
        G.add_node(obj["id"], label=obj["id"])  # remove caption for now, just show ID to avoid clutter.


    # Add edges
    for e in edges:
        G.add_edge(e["src"], e["dst"], label=e["relation"])

    pos = nx.spring_layout(G, k=1.2, iterations=50)

    plt.figure(figsize=(12, 8))

    # Node labels
    node_labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}

    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_size=1600, node_color="skyblue")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=25)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_color="red", font_size=8
    )

    plt.title("Scene Graph")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

    return G


def draw_objects_on_frame(rgb_frame, objects, font_scale=0.6, thickness=2):
    """
    Draw bounding boxes and captions on an RGB frame.

    objects: list of {bbox, caption, id, frame}
    """
    img = rgb_frame.copy()
    print("Drawing objects on frame:")
    for obj in objects:
        x, y, w, h = map(int, obj["bbox"])
        caption = obj["caption"]

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness)

        # Draw caption background
        cv2.rectangle(img, (x, y - 20), (x + len(caption) * 8, y), (0, 255, 0), -1)

        # Draw caption text
        cv2.putText(img, caption, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thickness // 2, cv2.LINE_AA)

    return img


def draw_relations_on_frame(rgb_frame, objects, edges,
                            arrow_color=(255, 0, 0),
                            text_color=(255, 0, 255),
                            thickness=2):
    """
    Draw lines/arrows between object centers, with relation text.
    """
    print("Drawing relations on frame...")
    img = rgb_frame.copy()

    # Create ID → object lookup
    obj_map = {obj["id"]: obj for obj in objects}

    for edge in edges:
        src = obj_map[edge["src"]]
        dst = obj_map[edge["dst"]]

        x1, y1 = map(int, src["center"])
        x2, y2 = map(int, dst["center"])

        # Draw arrow (A → B)
        cv2.arrowedLine(img, (x1, y1), (x2, y2), arrow_color, thickness)

        # Draw relation text at midpoint
        mid = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.putText(img, edge["relation"], mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    text_color, thickness, cv2.LINE_AA)

    return img


def visualize_full_pipeline(rgb, objects, edges):
    print("Visualizing full pipeline for one frame...")
    frame_boxes = draw_objects_on_frame(rgb, objects)
    frame_rel = draw_relations_on_frame(frame_boxes, objects, edges)

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.title("Objects + Relations")
    plt.imshow(frame_rel)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Scene Graph")
    draw_scene_graph(objects, edges, show=False)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_all_frames(result):
    print("Visualizing all frames...")
    frames = result["frames"]
    frame_objs = result["frame_objects"]
    edges = result["edges"]

    output_dir = result["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    for frame_idx, rgb in frames.items():

        objs = frame_objs[frame_idx]
        frame_edges = [e for e in edges if e["frame"] == frame_idx]

        # ---------------------------------------------------------------------
        # Show image with bounding boxes
        # ---------------------------------------------------------------------
        vis_img = draw_objects_on_frame(rgb, objs)
        #plt.figure(figsize=(8, 5))
        plt.figure(figsize=(20, 10))
        plt.imshow(vis_img)
        plt.axis("off")
        plt.title(f"Frame {frame_idx}: Objects")
        if output_dir:
            cv2.imwrite(
                f"{output_dir}/frame_{frame_idx}_objects.jpg",
                cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            )

        # ---------------------------------------------------------------------
        # Draw scene graph for this frame ONLY
        # ---------------------------------------------------------------------
        draw_scene_graph_per_frame(
            frame_idx,
            objs,
            frame_edges,
            save_path=f"{output_dir}/frame_{frame_idx}_scene_graph.jpg"
            if output_dir else None
        )
def draw_scene_graph_per_frame(frame_idx, objects, edges, save_path=None):
    """
    Draw a scene graph *only for one frame*.

    Parameters:
        frame_idx: int
        objects: list of object dicts for that frame
        edges: list of edge dicts for that frame
        save_path: optional path to save

    Each object must contain:
        - id
        - caption

    Each edge must contain:
        - src
        - dst
        - relation
    """

    G = nx.DiGraph()

    # ------------------------------------------
    # Add nodes for objects in this frame
    # ------------------------------------------
    for obj in objects:
        G.add_node(obj["id"], label=obj["id"]) # remove caption for now, just show ID to avoid clutter.
        #G.add_node(obj["id"], label=obj["caption"])


    # ------------------------------------------
    # Add edges for relations of this frame
    # ------------------------------------------
    for e in edges:
        G.add_edge(e["src"], e["dst"], label=e["relation"])

    # ------------------------------------------
    # Layout and drawing
    # ------------------------------------------
    pos = nx.spring_layout(G, k=1.4, iterations=50)

    plt.figure(figsize=(20, 15))

    node_labels = {n: G.nodes[n]['label'] for n in G.nodes()}
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}

    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="#85C1E9")
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=15)

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color="red",
        font_size=9
    )

    plt.title(f"Scene Graph — Frame {frame_idx}")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()

def draw_scene_graph_per_frame_SHK(frames_rgb, frame_objects, edges, out_dir="vis_frames_per_frame"):
    os.makedirs(out_dir, exist_ok=True)

    for frame_idx, rgb in frames_rgb.items():
        objs = frame_objects.get(frame_idx, [])
        frame_edges = [e for e in edges
                       if any(o["id"] == e["src"] for o in objs)
                       or any(o["id"] == e["dst"] for o in objs)]

        G = draw_scene_graph(objs, frame_edges, show=False)

        out_path = os.path.join(out_dir, f"scene_graph_{frame_idx:04d}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

