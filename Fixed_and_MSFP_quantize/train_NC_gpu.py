import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, TUDataset, Coauthor, Flickr
from torch_geometric.transforms import NormalizeFeatures

from quantised_gcn_layer import FixedGCNConv, MSFPGCNConv, bothGCNConv
from quantised_gcn_layer_super import CustomGCNConv
from quantised_sage_layer import CustomSAGEConv
from torch_geometric.nn import GCNConv
import numpy as np
import matplotlib.pyplot as plt
import os


import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import time
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table
#from pytorch_memlab import MemReporter
from thop import profile as pf
import quantised_gat_layer
from quantised_gat_layer import CustomGATConv
import argparse

def run_NC_training_loop(accuracy: list):

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #data
    #dataset = TUDataset(root='data/TUDataset', name='REDDIT-BINARY')
    #dataset = Flickr(root='data/Flickr')
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    #dataset = Planetoid(root='data/Planetoid', name='Citeseer', transform=NormalizeFeatures())
    #dataset = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())
    #dataset = Coauthor(root='data/Planetoid', name='Physics')
    data = dataset[0].to(device)  # Get the first graph object.

    #model
    class QGCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(1234567)
            self.conv1 = bothGCNConv(dataset.num_features, hidden_channels)
            self.conv2 = bothGCNConv(hidden_channels, dataset.num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x.to(device)
        
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden_channels', type=int, default=8)
    parser.add_argument('--heads', type=int, default=8)
    #parser.add_argument('--lr', type=float, default=0.005)
    #parser.add_argument('--epochs', type=int, default=200)
    #parser.add_argument('--wandb', action='store_true', help='Track experiment')
    args = parser.parse_args()

    class GAT(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, heads):
            super().__init__()
            self.conv1 = CustomGATConv(in_channels, hidden_channels, heads, dropout=0.6)
            # On the Pubmed dataset, use `heads` output heads in `conv2`.
            self.conv2 = CustomGATConv(hidden_channels * heads, out_channels, heads=1,
                                concat=False, dropout=0.6)

        def forward(self, x, edge_index):
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
            return x


    #model = QGCN(hidden_channels=16)
    #print(model)
    #Train and Test
    #model = QGCN(hidden_channels=16).to(device)
    model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
            args.heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss

    def test():
        model.eval()
        out = model(data.x.to(device), data.edge_index.to(device))
        pred = out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        return test_acc

    def print_gradients():
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f'{name} gradient: {param.grad}')

    def save_gradients(model, filename):
        # Open the file for writing
        with open(filename, 'a') as f:
            # Loop over all named parameters in the model
            for name, param in model.named_parameters():
                # Check if the parameter has a gradient
                if param.grad is not None:
                    # Write the name of the parameter and its gradient to the file
                    f.write(f'{name}:\n{param.grad}\n\n')

    #def save_model_weights_to_txt(model, filename):
    #    with open(filename, 'a') as f:
    #        f.write(f'Epoch {epoch}\n')
    #        for name, param in model.named_parameters():
    #            f.write(f'{name}\n')
    #            f.write(f'{param}\n')

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
    peak_memory_usage_all= []
    iterations =[]

    for epoch in range(1, 151):
        #torch.cuda.synchronize()
        #start_epoch = time.time()
        loss = train()
        #torch.cuda.synchronize()
        #end_epoch = time.time()
        #elapsed = end_epoch - start_epoch
        #times.append(elapsed)

        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if epoch % 1 == 0:
            #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            #save_gradients(model, 'gradients_MSFP.txt')
            #save_model_weights_to_txt(model, 'weights_MSFP.txt')
            # track memory usage
            current_memory_usage = torch.cuda.max_memory_allocated(device)
            peak_memory_usage = max(peak_memory_usage, current_memory_usage)
            # Collect memory usage statistics
            memory_stats = torch.cuda.memory_stats(device=device)
            peak_memory_usage_all.append(memory_stats["allocated_bytes.all.peak"])
            iterations.append(epoch)
            #total_time =  sum(times)
            
    print('#####Running Evaluation####')
    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')
    accuracy.append(test_acc)

""" def run_full_training_loop():
    peak_memory_usage = 0
    peak_memory_usage_all= []
    iterations =[]    
    with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available(), with_flops=True) as prof:

        for epoch in range(1, 151):
            #torch.cuda.synchronize()
            #start_epoch = time.time()
            loss = train()
            #torch.cuda.synchronize()
            #end_epoch = time.time()
            #elapsed = end_epoch - start_epoch
            #times.append(elapsed)

            #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            if epoch % 1 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
                #save_gradients(model, 'gradients_MSFP.txt')
                #save_model_weights_to_txt(model, 'weights_MSFP.txt')
                # track memory usage
                current_memory_usage = torch.cuda.max_memory_allocated(device)
                peak_memory_usage = max(peak_memory_usage, current_memory_usage)
                # Collect memory usage statistics
                memory_stats = torch.cuda.memory_stats(device=device)
                peak_memory_usage_all.append(memory_stats["allocated_bytes.all.peak"])
                iterations.append(epoch)
                #total_time =  sum(times)  
        print('#####Profiler Processing####')


    print('#####Running Evaluation####')
    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f"Peak GPU memory usage: {peak_memory_usage/(1024*1024)} MB")
    #print(f"Total GPU Time: {total_time} ")

    # get the model's state dictionary
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
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
    #print(f"Total memory reads: {profile.total_memory_read_events()}")
    #print(f"Total memory writes: {profile.total_memory_write_events()}")

    # Convert peak memory usage from bytes to megabytes
    peak_memory_usage_all_mb = [usage / (1024*1024) for usage in peak_memory_usage_all]
    # Plot the memory usage statistics
    #plt.plot(iterations, peak_memory_usage_all_mb)
    #plt.xlabel("Iteration")
    #plt.ylabel("Peak memory usage (mb)")
    #plt.show()

    # Print the top 10 memory writes and top 10 memory reads
    #print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))


    # Save profiler summary to a file
    #with open("profiler_summary.txt", "w") as f:
    #    f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage", row_limit=10)) """
