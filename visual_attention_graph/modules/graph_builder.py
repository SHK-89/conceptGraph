
import networkx as nx


class GraphBuilder:

    def build(self, all_scanpaths):

        G = nx.DiGraph()

        transitions = {}

        object_bboxes = {}

        # --------------------------------
        # collect nodes
        # --------------------------------

        for scanpath in all_scanpaths:

            for step in scanpath:

                obj_id = step["object_id"]
                bbox = step["bbox"]

                object_bboxes[obj_id] = bbox

        for obj_id, bbox in object_bboxes.items():

            G.add_node(obj_id, bbox=bbox)

        # --------------------------------
        # compute transitions
        # --------------------------------

        for scanpath in all_scanpaths:

            for i in range(len(scanpath)-1):

                src = scanpath[i]["object_id"]
                dst = scanpath[i+1]["object_id"]

                if src == dst:
                    continue

                key = (src, dst)

                transitions[key] = transitions.get(key, 0) + 1

        # --------------------------------
        # add edges
        # --------------------------------

        for (src, dst), weight in transitions.items():

            G.add_edge(src, dst, weight=weight)

        return G