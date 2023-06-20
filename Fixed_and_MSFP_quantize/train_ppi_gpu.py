#import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv

from quantised_gat_layer import CustomGATConv
from quantised_gcn_layer import FixedGCNConv, MSFPGCNConv, bothGCNConv
from quantised_gcn_layer_super import CustomGCNConv
from quantised_sage_layer import CustomSAGEConv
from quantised_gat_layer import CustomGATConv
from uniform_quantize_linear import FixedLinear_quantize_linear_layer, MSFPLinear_quantize_linear_layer
import json

def load_json_file(file_path):
    with open(file_path) as f:
        return json.load(f)
file_path_json = r'C:\Users\IBqai\Desktop\GNN\code\src\Fixed_and_MSFP_quantize\bitconfig.json'
bitconfig = load_json_file(file_path_json)



def run_ppi_training_loop(accuracy: list):

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
            self.conv1 = CustomGATConv(train_dataset.num_features, 256, heads=4)
            self.conv2 = CustomGATConv(4 * 256, 256, heads=4)
            self.conv3 = CustomGATConv(4 * 256, train_dataset.num_classes, heads=6,
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
            x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
            x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
            x = self.conv3(x, edge_index) + self.lin3(x)
            return x 

    """class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = CustomSAGEConv(train_dataset.num_features, 256)
            self.conv2 = CustomSAGEConv(256, 256)
            self.conv3 = CustomSAGEConv(256, train_dataset.num_classes,
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


    for epoch in range(1, 101):
        loss = train()
        val_f1 = test(val_loader)
        test_f1 = test(test_loader)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_f1:.4f}, '
            f'Test: {test_f1:.4f}')
    
    accuracy.append(test_f1)