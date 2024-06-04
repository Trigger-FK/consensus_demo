import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from Consensus_base.demo_consensus import network


def draw_network():
    cn = network()
    L = cn.Laplacian()
    D = np.diag(np.diag(L))
    G = nx.Graph(D - L)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight="bold")
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='c', width=2)

    plt.title("Network Topology")
    plt.savefig("Topology.png")
    plt.show()


draw_network()
