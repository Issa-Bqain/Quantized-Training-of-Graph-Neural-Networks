#import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv

import json

import quantised_gat_layer 
import quantised_gcn_layer 
import quantised_gcn_layer_super 
import quantised_sage_layer 
from uniform_quantize_linear import FixedLinear_quantize_linear_layer, MSFPLinear_quantize_linear_layer

from utils_run import reload_layers_DQ, change_fixed_bitwdith_DQ, change_msfp_bitwdith_DQ
import importlib


import numpy as np
import matplotlib.pyplot as plt
import os
import importlib


def load_json_file(file_path):
    with open(file_path) as f:
        return json.load(f)
file_path_json = r'C:\Users\IBqai\Desktop\GNN\code\src\Fixed_and_MSFP_quantize\bitconfig.json'
bitconfig = load_json_file(file_path_json)

current_avg = 0
previous_avg = 0


def run_ppi_DQ_training_loop(accuracy: list, DQ: bool, DQB: list, losses: list=[]):

    #path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
    #path = root='data/Planetoid'
    train_dataset = PPI(root='data/ppi', split='train')
    val_dataset = PPI(root='data/ppi', split='val')
    test_dataset = PPI(root='data/ppi', split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            reload_layers_DQ()
            self.conv1 = quantised_gat_layer.CustomGATConv(train_dataset.num_features, 256, heads=4)
            self.conv2 = quantised_gat_layer.CustomGATConv(4 * 256, 256, heads=4)
            self.conv3 = quantised_gat_layer.CustomGATConv(4 * 256, train_dataset.num_classes, heads=6,
                                concat=False)
            if bitconfig["qscheme"] == 'fixed':
                self.lin1 = FixedLinear_quantize_linear_layer(train_dataset.num_features, 4 * 256)
                self.lin2 = FixedLinear_quantize_linear_layer(4 * 256, 4 * 256)
                self.lin3 = FixedLinear_quantize_linear_layer(4 * 256, train_dataset.num_classes)
            if bitconfig["qscheme"] == 'msfp':
                self.lin1 = MSFPLinear_quantize_linear_layer(train_dataset.num_features, 4 * 256)
                self.lin2 = MSFPLinear_quantize_linear_layer(4 * 256, 4 * 256)
                self.lin3 = MSFPLinear_quantize_linear_layer(4 * 256, train_dataset.num_classes)
            if bitconfig["qscheme"] == 'none':
                self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
                self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
                self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)
        def forward(self, x, edge_index):
            reload_layers_DQ()
            x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
            x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
            x = self.conv3(x, edge_index) + self.lin3(x) 
            return x  

    """    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            reload_layers_DQ()
            self.conv1 = quantised_sage_layer.CustomSAGEConv(train_dataset.num_features, 256)
            self.conv2 = quantised_sage_layer.CustomSAGEConv(256, 256)
            self.conv3 = quantised_sage_layer.CustomSAGEConv(256, train_dataset.num_classes,
                                concat=False)
            if bitconfig["qscheme"] == 'fixed':
                self.lin1 = FixedLinear_quantize_linear_layer(train_dataset.num_features, 256)
                self.lin2 = FixedLinear_quantize_linear_layer(256, 256)
                self.lin3 = FixedLinear_quantize_linear_layer(256, train_dataset.num_classes)
            if bitconfig["qscheme"] == 'msfp':
                self.lin1 = MSFPLinear_quantize_linear_layer(train_dataset.num_features, 256)
                self.lin2 = MSFPLinear_quantize_linear_layer(256, 256)
                self.lin3 = MSFPLinear_quantize_linear_layer( 256, train_dataset.num_classes)
            if bitconfig["qscheme"] == 'none':
                self.lin1 = torch.nn.Linear(train_dataset.num_features, 256)
                self.lin2 = torch.nn.Linear(256, 256)
                self.lin3 = torch.nn.Linear( 256, train_dataset.num_classes)

        def forward(self, x, edge_index):
            reload_layers_DQ()
            x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
            x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
            x = self.conv3(x, edge_index) + self.lin3(x)
            return x  """ 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    loss_op = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


    def train():
        model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = loss_op(model(data.x, data.edge_index), data.y)
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def test(loader):
        model.eval()

        ys, preds = [], []
        for data in loader:
            ys.append(data.y)
            out = model(data.x.to(device), data.edge_index.to(device))
            preds.append((out > 0).float().cpu())

        y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
        return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

    current_avg = 0
    previous_avg = 0

    for epoch in range(1, 101):
        loss = train()
        val_f1 = test(val_loader)
        test_f1 = test(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_f1:.4f}, '
            f'Test: {test_f1:.4f}')
        
        loss_cpu = loss
        losses.append(loss_cpu)

        if epoch >= 11 and epoch % 5 == 1:
            current_avg = np.mean(losses[-5:])  # Calculate the mean of the last 5 losses
            previous_avg = np.mean(losses[-10:-5])  # Calculate the mean of the 5 losses before the last 5

        #if current_avg < 0.95 * previous_avg or current_avg > 1.02 * previous_avg :
        if current_avg > 0.95 * previous_avg:
            print("################MA threshold reached#################")
            if DQB[0]+1 <= DQB[1]:
                print("################bitwidth increased#################")
                DQB[0] = DQB[0]+1
                #change_fixed_bitwdith_DQ(DQB[0])
                change_msfp_bitwdith_DQ(DQB[0],8)
                reload_layers_DQ()
                importlib.reload(quantised_gcn_layer_super)
                importlib.reload(quantised_gcn_layer)
                
            else:
                print("################bitwidth is at max#################")


        """ if epoch == 25 and DQ == 1 :
            print("dynamic entered")
            #change_fixed_bitwdith_DQ(DQB[1])
            change_msfp_bitwdith_DQ(DQB[1],8)
            reload_layers_DQ()
            importlib.reload(quantised_gcn_layer_super)
            importlib.reload(quantised_gcn_layer)
        if epoch == 50 and DQ == 1 :
            print("dynamic entered")
            #change_fixed_bitwdith_DQ(DQB[2])
            change_msfp_bitwdith_DQ(DQB[2],8)
            reload_layers_DQ()
            importlib.reload(quantised_gcn_layer_super)
            importlib.reload(quantised_gcn_layer)
        if epoch == 75 and DQ == 1 :
            print("dynamic entered")
            #change_fixed_bitwdith_DQ(DQB[3])
            change_msfp_bitwdith_DQ(DQB[3],8)
            reload_layers_DQ()
            importlib.reload(quantised_gcn_layer_super)
            importlib.reload(quantised_gcn_layer) """
    
    accuracy.append(test_f1)