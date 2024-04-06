import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from demo import network


def draw_network():
    cn = network()
    L = cn.Laplacian()
    D = np.diag(np.diag(L))
    G = nx.Graph(D - L)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=300, font_size=16)

    plt.title("Network Topology")
    plt.savefig("Topology.png")


draw_network()
