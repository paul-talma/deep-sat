import os
from matplotlib import pyplot as plt
import networkx as nx
from pysat.solvers import Glucose4 as GC
from pysat.formula import CNF
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool

from tqdm import tqdm

# data
SAT_FOLDER = '/Users/paultalma/Programming/python_deep_sat/data/uf100-430'
UNSAT_FOLDER = '/Users/paultalma/Programming/python_deep_sat/data/uf100-430'


# determine max number of clauses in any formula
max_clauses = 0
for file in os.listdir(SAT_FOLDER):
    filepath = os.path.join(SAT_FOLDER, file)
    cnf = CNF(from_file=filepath)
    num_clauses = len(cnf.clauses)
    if num_clauses > max_clauses:
        max_clauses = num_clauses

# helper function
def flatten(lst):
    return [x for xs in lst for x in xs]

# graph building function
def build_graph(cnf, sat):
    G = nx.Graph(y=sat)
    edge_label = [0] * max_clauses
    # add edges between co-occurrent clauses
    for clause_number, clause in enumerate(cnf):
        elist = [
            (a, b)
            for (i, a) in enumerate(clause)
            for (j, b) in enumerate(clause)
            if i < j
        ]
        for (a, b) in elist:
            if (a, b) in G.edges:
                G[a][b]['vec'][clause_number] = 1
            else:
                edge_vec = edge_label.copy()
                edge_vec[clause_number] = 1
                G.add_edge(a, b, vec=edge_vec, neg='no')
        # G.add_edges_from(elist, negation='no')
    
    # add special edges for negations
    elist = []
    for a in flatten(cnf):
        for b in flatten(cnf):
            if a == -b and a > 0 and (a, b) not in elist:
                elist.append((a, b))
    
    # possibly faster
    # [
    #     (a, b) for a in flatten(cnf) for b in flatten(cnf) if a == -b and a > 0
    # ]
    G.add_edges_from(elist, neg='yes', vec=edge_label)

    # record which nodes are negations
    for node in G:
        G.nodes[node]['neg'] = int(node < 0)

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

# build graphs
sat_graph_list = []
unsat_graph_list = []

    # ... for sat instances
with open('/Users/paultalma/Programming/python_deep_sat/sat_log.txt', 'w') as log:
    log.write(SAT_FOLDER + "\n")

    for file in tqdm(os.listdir(SAT_FOLDER)[:5]):
        filepath = os.path.join(SAT_FOLDER, file)
        cnf = CNF(from_file=filepath)
        graph = build_graph(cnf, 1)
        sat_graph_list.append(graph)

    # ... for unsat instances
with open('/Users/paultalma/Programming/python_deep_sat/unsat_log.txt', 'w') as log:
    log.write(UNSAT_FOLDER + "\n")

    for file in tqdm(os.listdir(UNSAT_FOLDER)[:5]):
        filepath = os.path.join(UNSAT_FOLDER, file)
        cnf = CNF(from_file=filepath)
        graph = build_graph(cnf, 0)
        unsat_graph_list.append(graph)


# convert graphs from networkx to torch_geometric
print("Converting sat networkx graphs to tg graphs...")
sat_graph_list = [from_networkx(g) for g in tqdm(sat_graph_list)]
print("Converting unsat networkx graphs to tg graphs...")
unsat_graph_list = [from_networkx(g) for g in tqdm(unsat_graph_list)]

# merge sat and unsat instances
graph_list = sat_graph_list + unsat_graph_list

# debug
g = graph_list[0]
print(f"Graph: {g}")
print(f"Graph nodes: {g.x}")
print(f"Graph target: {g.y}")
print(f"Graph edges: {g.edge_index}")
print(f"Graph edges shape: {g.edge_index.shape}")
print(f"Graph edge attributes: {g.edge_attr}")

# split data into training and test
train_list = graph_list[:1600]
test_list = graph_list[1600:]
# sat_train = sat_graph_list[:800]
# sat_test = sat_graph_list[800:]
# unsat_train = unsat_graph_list[:800]
# unsat_test = unsat_graph_list[800:]

# batch data
print("Batching data...")
train_loader = DataLoader(train_list, batch_size=4, shuffle=True)
# sat_test_loader = DataLoader(sat_test, batch_size=64, shuffle=True)
# unsat_train_loader = DataLoader(unsat_train, batch_size=32, shuffle=True)
# unsat_test_loader = DataLoader(unsat_test, batch_size=32, shuffle=True)

# show batches
for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(f"data.x: {data.x}")
    print(f"targets: {data.y}")
    print(data)
    print()

# model class
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(-1, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

model = GNN(hidden_channels=64)
print(model)

# training function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass. x are nodes (?)
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

# main training loop
for epoch in range(1, 201):
    train()
    train_acc = test(train_loader)
    # test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')#, Test Acc: {test_acc:.4f}')