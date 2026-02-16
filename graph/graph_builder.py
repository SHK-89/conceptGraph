from geometry.geom2d import bbox_center, iou_2d, spatial_relation_heuristic

def build_2d_scene_graph_intervl(objects, captioner):
    """
    Creates a 2D scene graph using InternVL2-26B-Chat.
    """
    edges = []
    N = len(objects)

    for i in range(N):
        for j in range(i + 1, N):

            bA, bB = objects[i]["bbox"], objects[j]["bbox"]
            cA, cB = objects[i]["center"], objects[j]["center"]

            geom_rel = spatial_relation_heuristic(cA, cB)
            captionA = objects[i]["caption"]
            captionB = objects[j]["caption"]

            relation_text = captioner.relationship(
                captionA, captionB, geom_hint=geom_rel
            )
            print("--------------------------Relation Text:-------"+relation_text)
            edges.append({
                "src": i, "dst": j,
                "captionA": captionA,
                "captionB": captionB,
                "bbox_iou": float(iou_2d(bA, bB)),
                "geom_relation": geom_rel,
                "relation": relation_text
            })

    return edges
