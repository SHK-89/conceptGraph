import cv2
import numpy as np

from main import run_uvo_conceptgraphs_2d


def draw_bbox(img, bbox, color, thickness=2):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)


def draw_text(img, text, pos, color=(0, 255, 0), scale=0.6, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_arrow(img, p1, p2, color=(0, 255, 255), thickness=2):
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))
    cv2.arrowedLine(img, p1, p2, color, thickness, tipLength=0.03)


def visualize_scene_graph_2d(rgb, objects, edges, display=True, save_path=None):
    """
    Visualize 2D UVO ConceptGraphs scene graph on top of the RGB image.

    Parameters
    ----------
    rgb : np.ndarray
        RGB frame (H,W,3)
    objects : list of dict
        [{'bbox':..., 'center':..., 'caption':...}, ...]
    edges : list of dict
        [{'i': int, 'j': int, 'relation': "...", 'geometry_relation': "..."}]
    display : bool
        If True show using cv2.imshow
    save_path : str
        If provided, saves results to file
    """
    print(" visualizing 2D scene graph with {} objects and {} edges".format(len(objects), len(edges)))
    img = rgb.copy()[:, :, ::-1]  # convert RGB→BGR for OpenCV display
    colors = {}

    # assign random color for each object
    for i, obj in enumerate(objects):
        colors[i] = tuple(np.random.randint(50, 255, size=3).tolist())

    # Draw objects
    for i, obj in enumerate(objects):
        bbox = obj["bbox"]
        center = obj["center"]
        cap = obj["caption"]

        draw_bbox(img, bbox, colors[i])
        draw_text(img, f"ID {i}: {cap[:20]}", (int(center[0]), int(center[1])))

    # Draw edges (arrows)
    for e in edges:
        i, j = e["i"], e["j"]
        rel = e["relation"]
        geom = e["geometry_relation"]

        p1 = objects[i]["center"]
        p2 = objects[j]["center"]

        # arrow
        draw_arrow(img, p1, p2, color=(0, 255, 255))

        # text on midpoint
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        draw_text(img, rel[:25], (int(mid[0]), int(mid[1]) - 10), color=(0, 200, 255))
        draw_text(img, geom, (int(mid[0]), int(mid[1]) + 10), color=(255, 200, 0))

    if save_path:
        cv2.imwrite(save_path, img)

    if display:
        cv2.imshow("2D Scene Graph", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img





scene_graph = run_uvo_conceptgraphs_2d("vid.mp4", "vid.pkl")

frame = first_rgb_frame  # or load manually
visualize_scene_graph_2d(frame, scene_graph["objects"], scene_graph["edges"])