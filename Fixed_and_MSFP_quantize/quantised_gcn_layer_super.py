import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense.linear import Linear

from uniform_quantize_linear import MSFPLinear_quantize_linear_layer, FixedLinear_quantize_linear_layer

import json

#file_path = os.path.join(os.getcwd(), 'config.json')
file_path = r'C:\Users\IBqai\Desktop\GNN\code\src\Fixed_and_MSFP_quantize\bitconfig.json'
with open(file_path, 'r') as f:
    bitconfig = json.load(f)



class CustomGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)  # Call the parent class' constructor
        
        # Create a new self.lin with different attributes
        self.lin = torch.nn.Linear(in_channels, out_channels)  # Modify this line as per your requirements

        # Create a new self.lin with different attributes
        if bitconfig["qscheme"] == 'fixed':
            self.lin = FixedLinear_quantize_linear_layer(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
            print("############Using Fixed Quantisation############## ")

        if bitconfig["qscheme"] == 'msfp':
            self.lin = MSFPLinear_quantize_linear_layer(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
            print("############Using MSFP Quantisation############## ")

        if bitconfig["qscheme"] == 'none':
            self.lin = self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
            print("############Using no Quantisation############## ")

""" class CustomGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels):
        super(CustomGCNConv, self).__init__(in_channels, out_channels)  # Call the parent class' constructor
        
        # Create a new self.lin with different attributes
        if bitconfig["qscheme"] == 'fixed':
            self.lin = FixedLinear_quantize_linear_layer(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
            print("############Using Fixed Quantisation############## ")

        if bitconfig["qscheme"] == 'msfp':
            self.lin = MSFPLinear_quantize_linear_layer(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
            print("############Using MSFP Quantisation############## ")

        if bitconfig["qscheme"] == 'none':
            self.lin = self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
            print("############Using no Quantisation############## ")


class CustomGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels):
        super(CustomGCNConv, self).__init__(in_channels, out_channels)  # Call the parent class' constructor
        
        # Create a new self.lin with different attributes
        if bitconfig["qscheme"] == 'fixed':
            self.lin = FixedLinear_quantize_linear_layer(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
            print("############Using Fixed Quantisation############## ")

        if bitconfig["qscheme"] == 'msfp':
            self.lin = MSFPLinear_quantize_linear_layer(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
            print("############Using MSFP Quantisation############## ")

        if bitconfig["qscheme"] == 'none':
            self.lin = self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
            print("############Using no Quantisation############## ") """
            

            