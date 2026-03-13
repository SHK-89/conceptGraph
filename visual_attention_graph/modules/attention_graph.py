import networkx as nx


class AttentionGraphBuilder:

    def __init__(self):

        pass

    def build(self, transitions):

        G = nx.DiGraph()

        for (src, dst), weight in transitions:

            G.add_edge(src, dst, weight=weight)

        return G