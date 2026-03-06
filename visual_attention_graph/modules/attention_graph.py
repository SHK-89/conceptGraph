import networkx as nx


class AttentionGraphBuilder:

    def __init__(self, transition_matrix, object_ids):

        self.matrix = transition_matrix
        self.object_ids = object_ids

    def build_graph(self):

        G = nx.DiGraph()

        for obj in self.object_ids:
            G.add_node(obj)

        n = len(self.object_ids)

        for i in range(n):
            for j in range(n):

                weight = self.matrix[i, j]

                if weight > 0:
                    G.add_edge(
                        self.object_ids[i],
                        self.object_ids[j],
                        weight=weight
                    )

        return G