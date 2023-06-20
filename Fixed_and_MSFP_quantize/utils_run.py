import train_NC_gpu
import train_GC_gpu
import train_ppi_gpu
import train_LP_gpu
from train_NC_gpu import run_NC_training_loop
from train_GC_gpu import run_GC_training_loop
import uniform_quantize, uniform_quantize_linear, quantised_gcn_layer, quantised_gcn_layer_super, quantised_sage_layer, quantised_gat_layer
import json
import os
import subprocess
import time
import importlib

file_path = file_path = r'C:\Users\IBqai\Desktop\GNN\code\src\Fixed_and_MSFP_quantize\bitconfig.json'

import torch 
print(torch.__version__)

def change_fixed_bitwdith_DQ(bits):
    #access bitwdith values
    with open(file_path, 'r') as f:
        data = json.load(f)
        data['xbits'] = bits
        data['qscheme'] = 'fixed'
    with open(file_path, 'w') as f:
        json.dump(data, f)  

def change_msfp_bitwdith_DQ(mantissap1, exponent_width):    
    #access bitwdith values
    with open(file_path, 'r') as f:
        data = json.load(f)
        data['xmantissap1'] = mantissap1
        data['xexponent_width'] = exponent_width
        data['qscheme'] = 'msfp'
    with open(file_path, 'w') as f:
        json.dump(data, f) 

def change_to_normal_DQ():    
    #access bitwdith values
    with open(file_path, 'r') as f:
        data = json.load(f)
        data['qscheme'] = 'none'
    with open(file_path, 'w') as f:
        json.dump(data, f) 

def change_to_normal2_DQ():
    with open(file_path, 'r+') as f:
        data = json.load(f)
        data['qscheme'] = 'none' # <--- add id value.
        f.seek(0)        # <--- should reset file position to the beginning.
        json.dump(data, f, )
        f.truncate()     # remove remaining part

def reload_layers_DQ():
    importlib.reload(train_NC_gpu)
    importlib.reload(uniform_quantize)
    importlib.reload(uniform_quantize_linear)
    importlib.reload(quantised_gcn_layer)
    importlib.reload(quantised_gcn_layer_super)
    importlib.reload(quantised_sage_layer)
    importlib.reload(quantised_gat_layer)

    importlib.reload(train_GC_gpu)
    importlib.reload(uniform_quantize)
    importlib.reload(uniform_quantize_linear)
    importlib.reload(quantised_gcn_layer)
    importlib.reload(quantised_gcn_layer_super)
    importlib.reload(quantised_sage_layer)
    importlib.reload(quantised_gat_layer)

    importlib.reload(train_LP_gpu)
    importlib.reload(uniform_quantize)
    importlib.reload(uniform_quantize_linear)
    importlib.reload(quantised_gcn_layer)
    importlib.reload(quantised_gcn_layer_super)
    importlib.reload(quantised_sage_layer)
    importlib.reload(quantised_gat_layer)

    importlib.reload(train_ppi_gpu)
    importlib.reload(uniform_quantize)
    importlib.reload(uniform_quantize_linear)
    importlib.reload(quantised_gcn_layer)
    importlib.reload(quantised_gcn_layer_super)
    importlib.reload(quantised_sage_layer)
    importlib.reload(quantised_gat_layer)