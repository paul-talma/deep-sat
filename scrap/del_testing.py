from matplotlib import pyplot as plt
from formula_to_graph import build_graph, graph_statistics


form = [[0, 1, 2], [3, 4, 5], [2, 3, 6]]

data = build_graph(form)

graph_statistics(data)
