import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt



ANA, area_num = 0, 77
df = pd.read_csv("./UrbanKG/CHI/triplets_CHI.txt", sep=" ", header=None)
entity2id = pd.read_csv("./UrbanKG/CHI/entity2id_CHI.txt", sep=" ", header=None)

class Graph_Matrix:
    """
    Adjacency Matrix
    """
    def __init__(self, vertices=[], matrix=[]):
        """

        :param vertices:a dict with vertex id and index of matrix , such as {vertex:index}
        :param matrix: a matrix
        """
        self.matrix = matrix
        self.edges_dict = {}  # {(tail, head):weight}
        self.edges_array = []  # (tail, head, weight)
        self.vertices = vertices
        self.num_edges = 0

        # if provide adjacency matrix then create the edges list
        if len(matrix) > 0:
            if len(vertices) != len(matrix):
                raise IndexError
            self.edges = self.getAllEdges()
            self.num_edges = len(self.edges)

        # if do not provide a adjacency matrix, but provide the vertices list, build a matrix with 0
        elif len(vertices) > 0:
            self.matrix = [[0 for col in range(len(vertices))] for row in range(len(vertices))]
        self.num_vertices = len(self.matrix)

    def isOutRange(self, x):
        try:
            if x >= self.num_vertices or x <= 0:
                raise IndexError
        except IndexError:
            print("节点下标出界")

    def isEmpty(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices == 0

    def add_vertex(self, key):
        if key not in self.vertices:
            self.vertices[key] = len(self.vertices) + 1

        # add a vertex mean add a row and a column
        # add a column for every row
        for i in range(self.getVerticesNumbers()):
            self.matrix[i].append(0)

        self.num_vertices += 1

        nRow = [0] * self.num_vertices
        self.matrix.append(nRow)

    def getVertex(self, key):
        pass

    def add_edges_from_list(self, edges_list):  # edges_list : [(tail, head, weight),()]
        for i in range(len(edges_list)):
            self.add_edge(edges_list[i][0], edges_list[i][1], edges_list[i][2], )

    def add_edge(self, tail, head, cost=0):
        # if self.vertices.index(tail) >= 0:
        #   self.addVertex(tail)
        if tail not in self.vertices:
            self.add_vertex(tail)
        # if self.vertices.index(head) >= 0:
        #   self.addVertex(head)
        if head not in self.vertices:
            self.add_vertex(head)

        # for directory matrix
        self.matrix[self.vertices.index(tail)][self.vertices.index(head)] = cost
        # for non-directory matrix
        # self.matrix[self.vertices.index(fromV)][self.vertices.index(toV)] = \
        #   self.matrix[self.vertices.index(toV)][self.vertices.index(fromV)] = cost

        self.edges_dict[(tail, head)] = cost
        self.edges_array.append((tail, head, cost))
        self.num_edges = len(self.edges_dict)

    def getEdges(self, V):
        pass

    def getVerticesNumbers(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices

    def getAllVertices(self):
        return self.vertices

    def getAllEdges(self):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if 0 < self.matrix[i][j] < float('inf'):
                    self.edges_dict[self.vertices[i], self.vertices[j]] = self.matrix[i][j]
                    self.edges_array.append([self.vertices[i], self.vertices[j], self.matrix[i][j]])

        return self.edges_array

    def __repr__(self):
        return str(''.join(str(i) for i in self.matrix))

    def to_do_vertex(self, i):
        print('vertex: %s' % (self.vertices[i]))

    def to_do_edge(self, w, k):
        print('edge tail: %s, edge head: %s, weight: %s' % (self.vertices[w], self.vertices[k], str(self.matrix[w][k])))


def draw_directed_graph(my_graph):
    G = nx.DiGraph()  # 建立一个空的无向图G
    for node in my_graph.vertices:
        G.add_node(str(node))
    G.add_weighted_edges_from(my_graph.edges_array)

    print("nodes:", G.nodes())  # 输出全部的节点
    print("edges:", G.edges())  # 输出全部的边
    print("number of edges:", G.number_of_edges())  # 输出边的数量
    nx.draw(G, with_labels=True)
    plt.savefig("directed_graph.png")
    plt.show()


def get_area_id(area: str) -> int:
    return int(area.split("/")[1])

# def create_directed_graph_from_edges(edge_list):
#     nodes = ["Area/" + str(_) for _ in range(area_num + 1)]
#     my_graph = Graph_Matrix(nodes)
#     my_graph.add_edges_from_list(edge_list)
#     print(my_graph)
#     return my_graph
#
# def draw_directed_graph(my_graph):
#     G = nx.DiGraph()  # 建立一个空的无向图G
#     for node in my_graph.vertices:
#         G.add_node(str(node))
#     G.add_weighted_edges_from(my_graph.edges_array)
#
#     print("nodes:", G.nodes())  # 输出全部的节点
#     print("edges:", G.edges())  # 输出全部的边
#     print("number of edges:", G.number_of_edges())  # 输出边的数量
#     nx.draw(G, with_labels=True)
#     plt.savefig("directed_graph.png")
#     plt.show()


id2entity = {}
for _e, _i in entity2id.values:
    id2entity[_i] = _e
area = ["" + str(_) for _ in range(77)]
area_graph = [[0 for _i in range(area_num)] for _ in range(area_num)]
edges = []
for _h, _r, _t in df.values:
    if _r == ANA:
        # edges.append((id2entity[_h], id2entity[_t], 1))
        _h, _t = get_area_id(id2entity[_h]), get_area_id(id2entity[_t])
        area_graph[_h-1][_t-1] = 1

g = Graph_Matrix(vertices=area, matrix=area_graph)

draw_directed_graph(g)




