import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset, MNISTSuperpixels
from torch_geometric.transforms import NormalizeFeatures
from torch.nn import Linear

from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from torch_geometric.transforms import NormalizeFeatures
from quantised_gcn_layer import FixedGCNConv, MSFPGCNConv, bothGCNConv
from quantised_gcn_layer_super import CustomGCNConv
from quantised_sage_layer import CustomSAGEConv
from quantised_gat_layer import CustomGATConv
from uniform_quantize_linear import FixedLinear_quantize_linear_layer, MSFPLinear_quantize_linear_layer
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
import matplotlib.pyplot as plt
import os
import math
import json
import quantised_gat_layer
file_path_json = r'C:\Users\IBqai\Desktop\GNN\code\src\Fixed_and_MSFP_quantize\bitconfig.json'


def load_json_file(file_path):
    with open(file_path) as f:
        return json.load(f)

bitconfig = load_json_file(file_path_json)


def run_GC_training_loop(accuracy: list):

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #data
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    #dataset = TUDataset(root='data/TUDataset', name='ENZYMES', transform= NormalizeFeatures)
    #dataset = GNNBenchmarkDataset(root='data/TUDataset', name='CIFAR10')
    #dataset = MNISTSuperpixels(root='data/MNIST')  
    print(f'Number of graphs: {len(dataset)}')
    data = dataset[0].to(device)   # Get the first graph object.
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    datasplit = math.floor(0.8* len(dataset))
    print(datasplit)
    train_dataset = dataset[:datasplit]
    test_dataset = dataset[datasplit:]
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    #batching, currently none
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    #train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    #kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}



    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()

    #model

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            #torch.manual_seed(12345)
            self.conv1 = CustomGATConv(dataset.num_node_features, hidden_channels)
            self.conv2 = CustomGATConv(hidden_channels, hidden_channels)
            self.conv3 = CustomGATConv(hidden_channels, hidden_channels)

            if bitconfig["qscheme"] == 'fixed':
                self.lin = FixedLinear_quantize_linear_layer(hidden_channels, dataset.num_classes)
            if bitconfig["qscheme"] == 'msfp':
                self.lin = MSFPLinear_quantize_linear_layer(hidden_channels, dataset.num_classes)
            if bitconfig["qscheme"] == 'none':
                self.lin = self.lin = Linear(hidden_channels, dataset.num_classes)
                        
        def forward(self, x, edge_index, batch):
            # 1. Obtain node embeddings 
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)

            # 2. Readout layer
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

            # 3. Apply a final classifier
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin(x)
            
            return x.to(device)



    #train and evaluate

    model = GCN(hidden_channels=64).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            #data= data.to(device)
            out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)) # Perform a single forward pass.
            loss = criterion(out, data.y.to(device))  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y.to(device)).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    def get_model_weights_size(model):
        """
        Calculates the size of the weights of a PyTorch model in MB.
        
        Args:
            model: A PyTorch model object.
        
        Returns:
            The size of the model's weights in MB.
        """
        num_params = sum(p.numel() for p in model.parameters())
        size_mb = num_params * 4 / (1024 ** 2)  # assuming 4 bytes per parameter element
        return size_mb


    # initialize peak memory usage
    peak_memory_usage = 0
    peak_memory_usage_all = 0



    # initialize peak memory usage
    peak_memory_usage = 0
    peak_memory_usage_all= []
    iterations =[]

    #with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available(), with_flops=True) as prof:
    for epoch in range(1, 251):
        train()
        train_acc = test(train_loader)
        #test_acc = test(test_loader)
        #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
        # track memory usage
        current_memory_usage = torch.cuda.max_memory_allocated(device)
        peak_memory_usage = max(peak_memory_usage, current_memory_usage)
        # Collect memory usage statistics
        memory_stats = torch.cuda.memory_stats(device=device)
        peak_memory_usage_all.append(memory_stats["allocated_bytes.all.peak"])
        iterations.append(epoch)
#print('#####Profiler Processing####')


#print('#####Running Evaluation####')

    test_acc = test(test_loader)
    print(f'Test Accuracy: {test_acc:.4f}')
    accuracy.append(test_acc)
    #print(f"Peak GPU memory usage: {peak_memory_usage/(1024*1024)} MB")



""" # get the model's state dictionary
state_dict = model.state_dict()
# save the state dictionary to a temporary file
tmp_file = "tmp.pt"
torch.save(state_dict, tmp_file)
# get the size of the file in megabytes
size_mb = os.path.getsize(tmp_file) / (1024 * 1024)
# remove the temporary file
#os.remove(tmp_file)
print(f"The size of the model's weights is {size_mb:.2f} MB.")
size_mb_2 = get_model_weights_size(model)
print(f"The size of the model's weights_method_2 is {size_mb:.2f} MB.")
# Print the top 10 memory writes and top 10 memory reads
#print(prof.key_averages().table(row_limit=300))

#print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
#print(f"Total memory reads: {profile.total_memory_read_events()}")
#print(f"Total memory writes: {profile.total_memory_write_events()}")

# Convert peak memory usage from bytes to megabytes
peak_memory_usage_all_mb = [usage / (1024*1024) for usage in peak_memory_usage_all]
# Plot the memory usage statistics
plt.plot(iterations, peak_memory_usage_all_mb)
plt.xlabel("Iteration")
plt.ylabel("Peak memory usage (mb)")
plt.show() """