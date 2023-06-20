from torch_geometric.datasets import Planetoid, Reddit, Amazon, GNNBenchmarkDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import TUDataset


# Load your dataset
dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#dataset = Planetoid(root='data/Planetoid', name='Cora')
#dataset = Planetoid(root='data/Planetoid', name='Citeseer', transform=NormalizeFeatures())
#dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())
#dataset = Reddit(root='/path/to/dataset') #too big
#dataset = TUDataset(root='data/TUDataset/Reddit', name='REDDIT-BINARY')
#dataset = Amazon(root='data/Amazon', name='Computers')
#dataset = TUDataset(root='data/TUDataset', name='ENZYMES')
#dataset = GNNBenchmarkDataset(root='data/TUDataset', name='CIFAR10') 

# Calculate total number of nodes, edges, features, graphs, and classes
total_nodes = 0
total_edges = 0
total_features = dataset.num_features
total_graphs = len(dataset)
total_classes = dataset.num_classes

for data in dataset:
    total_nodes += data.num_nodes
    total_edges += data.num_edges

print("Total number of graphs:", total_graphs)
print("Total number of nodes:", total_nodes)
print("Total number of edges:", total_edges)
print("Total number of features (per node):", total_features)
print("Total number of classes:", total_classes)


""" 
if bitconfig["qscheme"] == 'fixed':
    self.lin = 

if bitconfig["qscheme"] == 'msfp':
    self.lin = 

if bitconfig["qscheme"] == 'none':
    self.lin =  """