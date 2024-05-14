import networkx as nx
from matplotlib import pyplot as plt


def extract_formula(filename):
    with open(filename, "r") as fid:
        formula = [
            [int(num) for num in line.strip().strip(" 0").split()]
            for line in fid
            if (not line.startswith(("c", "p", "%", "0")))
        ]
    return formula[:-1]


def flatten(lst):
    return [x for xs in lst for x in xs]

def build_graph(form):
    G = nx.MultiGraph()
    
    # add edges between co-occurrent clauses
    for clause_number, clause in enumerate(form):
        elist = [
            (a, b)
            for (i, a) in enumerate(clause)
            for (j, b) in enumerate(clause)
            if i < j
        ]
        G.add_edges_from(elist, clause=(clause_number + 1), neg='no')
    
    # add special edges for negations
    elist = []
    for idx, a in enumerate(flatten(form)):
        for b in flatten(form)[idx:]:
            if a == -b and a > 0 and (a, b) not in elist:
                elist.append((a, b))
    
    # possibly faster
    # [
    #     (a, b) for a in flatten(form) for b in flatten(form) if a == -b and a > 0
    # ]
    G.add_edges_from(elist, neg='yes')
    return G


def visualize(G):
    options = {
        "font_size": 36,
        "node_size": 3000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 5,
        "width": 5,
    }
    # nx.draw_networkx(G, **options)
    nx.draw(G, with_labels=True)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
