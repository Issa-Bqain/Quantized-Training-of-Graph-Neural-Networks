import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from quantised_gcn_layer import FixedGCNConv
from torch_geometric.nn import GCNConv
import numpy as np

#data
#dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
dataset = Planetoid(root='data/Planetoid', name='Citeseer', transform=NormalizeFeatures())
data = dataset[0]  # Get the first graph object.
#model
class QGCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = FixedGCNConv(dataset.num_features, hidden_channels)
        self.conv2 = FixedGCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = QGCN(hidden_channels=16)
print(model)
#Train and Test
model = QGCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
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

def save_model_weights_to_txt(model, filename):
    with open(filename, 'a') as f:
        f.write(f'Epoch {epoch}\n')
        for name, param in model.named_parameters():
            f.write(f'{name}\n')
            f.write(f'{param}\n')


for epoch in range(1, 101):
    loss = train()
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch % 1 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        save_gradients(model, 'gradients_MSFP.txt')
        save_model_weights_to_txt(model, 'weights_MSFP.txt')


test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')
