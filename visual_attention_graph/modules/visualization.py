import matplotlib.pyplot as plt


class GraphVisualizer:

    def draw(self, G):

        pos = nx.spring_layout(G)

        weights = nx.get_edge_attributes(G, 'weight')

        nx.draw(G, pos, with_labels=True, node_size=1500)

        nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)

        plt.show()