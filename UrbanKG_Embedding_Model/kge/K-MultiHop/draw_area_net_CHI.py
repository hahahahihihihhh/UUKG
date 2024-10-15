import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from utils.graph import Graph_Matrix

ANA, area_num = 1, 77
entity2id = pd.read_csv("../../data/CHI/entity2id_CHI.txt", sep=" ", header=None)

def draw_directed_graph(my_graph):
    G = nx.DiGraph()  # 建立一个空的无向图G
    for node in my_graph.vertices:
        G.add_node(str(node))
    G.add_weighted_edges_from(my_graph.edges_array)
    print("nodes:", G.nodes())  # 输出全部的节点
    print("edges:", G.edges())  # 输出全部的边
    print("number of edges:", G.number_of_edges())  # 输出边的数量
    nx.draw(G, with_labels=True)
    plt.savefig("area_net_CHI.png")
    plt.show()

def load_obj(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)

def get_area_id(area: str) -> int:
    return int(area.split("/")[1])

def main():
    id2entity = {}
    for _e, _i in entity2id.values:
        id2entity[_i] = _e

    area = ["" + str(_) for _ in range(77)]
    area_graph = load_obj('area_graph_CHI.pkl')
    g = Graph_Matrix(vertices=area, matrix=area_graph)
    draw_directed_graph(g)

if __name__ == '__main__':
    main()




