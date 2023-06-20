import train_NC_gpu
import train_GC_gpu
import train_LP_gpu
import train_ppi_gpu
from train_NC_gpu import run_NC_training_loop
from train_GC_gpu import run_GC_training_loop
from train_LP_gpu import run_LP_training_loop
from train_ppi_gpu import run_ppi_training_loop
#from train_ppi import run_ppi_training_loop
from dynamic_quantize_test import run_NC_DQ_training_loop
from dynamic_GC_gpu import run_DQ_GC_training_loop
from dynamic_LP_gpu import run_LP_DQ_training_loop
from dynamic_ppi_gpu import run_ppi_DQ_training_loop
import uniform_quantize, uniform_quantize_linear, quantised_gcn_layer, quantised_gcn_layer_super, quantised_sage_layer, quantised_gat_layer
import json
import os
import subprocess
import time
import importlib

import matplotlib.pyplot as plt

file_path = file_path = r'C:\Users\IBqai\Desktop\GNN\code\src\Fixed_and_MSFP_quantize\bitconfig.json'
print(file_path)


def load_json_file(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def change_fixed_bitwdith(bits):
    #access bitwdith values
    with open(file_path, 'r') as f:
        data = json.load(f)
        data['xbits'] = bits
        data['qscheme'] = 'fixed'
    with open(file_path, 'w') as f:
        json.dump(data, f)  

def change_msfp_bitwdith(mantissap1, exponent_width):    
    #access bitwdith values
    with open(file_path, 'r') as f:
        data = json.load(f)
        data['xmantissap1'] = mantissap1
        data['xexponent_width'] = exponent_width
        data['qscheme'] = 'msfp'
    with open(file_path, 'w') as f:
        json.dump(data, f) 

def change_to_normal():    
    #access bitwdith values
    with open(file_path, 'r') as f:
        data = json.load(f)
        data['qscheme'] = 'none'
    with open(file_path, 'w') as f:
        json.dump(data, f) 

def change_to_normal2():
    with open(file_path, 'r+') as f:
        data = json.load(f)
        data['qscheme'] = 'none' # <--- add id value.
        f.seek(0)        # <--- should reset file position to the beginning.
        json.dump(data, f, )
        f.truncate()     # remove remaining part

def reload_layers():
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

accuracy_list = []
accuracy_list_fixed = []
accuracy_list_msfp = []
accuracy_list_dq = []
test = []

#change_msfp_bitwdith(4,8)
#importlib.reload(train_NC_gpu)
#importlib.reload(uniform_quantize)
#importlib.reload(uniform_quantize_linear)
#importlib.reload(quantised_gcn_layer)
#importlib.reload(quantised_gcn_layer_super)
#run_NC_training_loop(accuracy_list)
#print(accuracy_list)


def train_NC():
    change_to_normal()
    reload_layers()
    run_NC_training_loop(accuracy_list)
    print(accuracy_list)


    for i in range(1,5):
        change_fixed_bitwdith(i*2)
        reload_layers()
        run_NC_training_loop(accuracy_list_fixed)
        print(accuracy_list_fixed)
    print(accuracy_list_fixed)

    for i in range(4,9):
        change_msfp_bitwdith(i,8)
        reload_layers()
        run_NC_training_loop(accuracy_list_msfp)
        print(accuracy_list_msfp) 

    print("finished")
    print(accuracy_list)
    print(accuracy_list_fixed)   
    print(accuracy_list_msfp)

def train_GC():
    change_to_normal()
    reload_layers()
    run_GC_training_loop(accuracy_list)
    print(accuracy_list)


    for i in range(1,5):
        change_fixed_bitwdith(i*2)
        reload_layers()
        run_GC_training_loop(accuracy_list_fixed)
        print(accuracy_list_fixed)
    print(accuracy_list_fixed)

    for i in range(4,9):
        change_msfp_bitwdith(i,8)
        reload_layers()
        run_GC_training_loop(accuracy_list_msfp)
        print(accuracy_list_msfp) 

    print("finished")
    print(accuracy_list)
    print(accuracy_list_fixed)   
    print(accuracy_list_msfp)

def train_LP():
    change_to_normal()
    reload_layers()
    run_LP_training_loop(accuracy_list)
    print(accuracy_list)


    for i in range(1,5):
        change_fixed_bitwdith(i*2)
        reload_layers()
        run_LP_training_loop(accuracy_list_fixed)
        print(accuracy_list_fixed)
    print(accuracy_list_fixed)

    for i in range(4,9):
        change_msfp_bitwdith(i,8)
        reload_layers()
        run_LP_training_loop(accuracy_list_msfp)
        print(accuracy_list_msfp) 

    print("finished")
    print(accuracy_list)
    print(accuracy_list_fixed)   
    print(accuracy_list_msfp)



def train_ppi():
    change_to_normal()
    reload_layers()
    run_ppi_training_loop(accuracy_list)
    print(accuracy_list)


    for i in range(1,5):
        change_fixed_bitwdith(i*2)
        reload_layers()
        run_ppi_training_loop(accuracy_list_fixed)
        print(accuracy_list_fixed)
    print(accuracy_list_fixed)

    for i in range(4,9):
        change_msfp_bitwdith(i,8)
        reload_layers()
        run_ppi_training_loop(accuracy_list_msfp)
        print(accuracy_list_msfp) 

    print("finished")
    print(accuracy_list)
    print(accuracy_list_fixed)   
    print(accuracy_list_msfp)





#train_NC()
#train_LP()
#train_GC()
#train_ppi()






#DQB = [8, 8, 8, 8]

#DQB = [2, 8, 8, 16]
#DQB = [16,8,8,2]
#DQB = [2, 8, 16, 32]
#DQB = [6, 8, 16, 32]
#DQB = [6, 8, 12, 16]

#DQB = [2,2,2,16]
#DQB =[4,2,2,16]
#DQB =[4,4,4,16]
#DQB =[16,8,4,4]
#DQB =[8,4,4,16]
#DQB =[8,8,8,16]
#DQB =[16,4,4,16]
#DQB =[16,8,8,16]

#DQB =[2,2,2,2]
#DQB =[32,32,32,32]
#DQB =[16,8,8,16]
#DQB =[16,4,4,16]
#DQB =[8,4,4,8]

loss1 = []
loss2 = []
loss3 = []
loss4 = []
loss5 = []
loss6 = []


#run_DQ_GC_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB)
#run_LP_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB)
#run_ppi_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB)
#print(accuracy_list_dq)



#####################################
#####################################
#####################################

""" DQB = [2, 8, 8, 16]
#QB= [2,2,2,2]
#DQB = [2,8]
#DQB = [2,4]
#DQB = [4,4]
#DQB = [2,2]
change_fixed_bitwdith(DQB[0])
#change_msfp_bitwdith(DQB[0],8)
Dynamic_quantisation = 1
run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss1)
print(accuracy_list_dq)

DQB = [16, 8, 8, 2]
#DQB= [3,3,3,3]
#DQB = [4,8]
#DQB = [2,6]
#DQB = [5,5]
#DQB = [4,4]
change_fixed_bitwdith(DQB[0])
#change_msfp_bitwdith(DQB[0],8)
Dynamic_quantisation = 1
run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss2)
print(accuracy_list_dq)

DQB = [16, 2, 2, 8]
#DQB = [2,3,3,3]
#DQB = [2,16]
#DQB = [3,5]
#DQB = [6,6]
#DQB = [6,6]
change_fixed_bitwdith(DQB[0])
#change_msfp_bitwdith(DQB[0],8)
Dynamic_quantisation = 1
run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss3)
print(accuracy_list_dq)

DQB = [16, 4, 4, 8]
#DQB = [3,3,4,4]
#DQB = [7,7]
#DQB = [8,8]
change_fixed_bitwdith(DQB[0])
#change_msfp_bitwdith(DQB[0],8)
Dynamic_quantisation = 1
run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss4)
print(accuracy_list_dq)



DQB = [8, 4, 4, 8]
#DQB = [2,3,3,4]
#DQB = [3,6]
#DQB = [8,8]
#DQB = [16,16]
change_fixed_bitwdith(DQB[0])
#change_msfp_bitwdith(DQB[0],8)
Dynamic_quantisation = 1
run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss5)
print(accuracy_list_dq)


# Plotting the loss values
plt.plot(range(1, 151), loss1, color='red', label='SDQF = [2, 8, 8, 16]')
plt.plot(range(1, 151), loss2, color='blue',label='SDQF = [16, 8, 8, 2]')
plt.plot(range(1, 151), loss3, color='green',label='SDQF = [16, 2, 2, 8]')
plt.plot(range(1, 151), loss4, color='purple',label='SDQF = [16, 4, 4, 8]')
plt.plot(range(1, 151), loss5, color='magenta',label='SDQF = [8, 4, 4, 8]')
plt.xlabel('Epoch', fontsize=30)
plt.ylabel('Loss', fontsize=30)
plt.title('Validation Loss vs Epochs using fixed point quantization with a static dynamic quantization scheme', fontsize=30)
plt.legend(fontsize=28)
plt.show() """

#####################################
#####################################
#####################################


#change_fixed_bitwdith(32)
#reload_layers()
#run_NC_training_loop(accuracy_list_fixed)
#print(accuracy_list_fixed)

""" DQB = [16, 8, 8, 2]
change_fixed_bitwdith(DQB[0])
Dynamic_quantisation = 1
run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss2)
print(accuracy_list_dq)

DQB = [16, 2, 2, 8]
change_fixed_bitwdith(DQB[0])
Dynamic_quantisation = 1
run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss3)
print(accuracy_list_dq)

DQB = [16, 4, 4, 8]
change_fixed_bitwdith(DQB[0])
Dynamic_quantisation = 1
run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss4)
print(accuracy_list_dq)


DQB = [8, 4, 4, 8]
change_fixed_bitwdith(DQB[0])
Dynamic_quantisation = 1
run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss5)
print(accuracy_list_dq)

DQB = [8, 8, 8, 8]
change_fixed_bitwdith(DQB[0])
Dynamic_quantisation = 1
run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB)
print(accuracy_list_dq)


# Plotting the loss values
plt.plot(range(1, 151), loss1, color='red', label='[2, 8, 8, 16]')
plt.plot(range(1, 151), loss2, color='blue',label='[16, 8, 8, 2]')
plt.plot(range(1, 151), loss3, color='green',label='[16, 2, 2, 8]')
plt.plot(range(1, 151), loss4, color='purple',label='[16, 4, 4, 8]')
plt.plot(range(1, 151), loss5, color='orange',label='[8, 4, 4, 8]')
#plt.plot(range(1, 151), loss6, color='magenta',label='[16, 4, 4, 16]')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses vs Epoch using a fixed-point static dynamic quantization scheme')
plt.legend()
plt.show() """




#DQB =[2,2,2,2]
#DQB =[3,3,3,3]
#DQB =[2,2,3,3]
#DQB =[2,3,3,4]
#DQB =[2,2,4,4]
#DQB =[3,3,4,4]
#DQB =[2,3,3,4]
#DQB =[4,2,2,4]
#DQB =[4,3,3,4]
#DQB =[8,8,8,8]
#DQB =[3,2,2,3]
#DQB =[2,3,3,3]
#DQB =[3,4,5,6]
#DQB= [5,4,4,5]
#DQB =[6,5,5,6]


#DQB =[3,3,4,4]
#DQB =[3,4,5,6]
#DQB =[4,3,3,4]



#change_msfp_bitwdith(DQB[0], 8)
#reload_layers()
#Dynamic_quantisation = 1
#run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB)
#run_DQ_GC_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB)
#run_LP_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB)
#run_ppi_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB)
#print(accuracy_list_dq)


#####################################
#####################################
#####################################



#DQB = [2,8]
DQB = [2,4]
#change_fixed_bitwdith(DQB[0])
change_msfp_bitwdith(DQB[0],8)
Dynamic_quantisation = 1
#run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss1)
run_DQ_GC_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB)
#run_LP_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss1)
#run_ppi_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss1)
print(accuracy_list_dq)


#DQB = [4,8]
DQB = [2,6]
#change_fixed_bitwdith(DQB[0])
change_msfp_bitwdith(DQB[0],8)
Dynamic_quantisation = 1
#run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss1)
run_DQ_GC_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB)
#run_LP_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss2)
#run_ppi_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss2)
print(accuracy_list_dq)

#DQB = [2,16]
DQB = [3,5]
#change_fixed_bitwdith(DQB[0])
change_msfp_bitwdith(DQB[0],8)
Dynamic_quantisation = 1
#run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss1)
run_DQ_GC_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB)
#run_LP_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss3)
#run_ppi_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss3)
print(accuracy_list_dq)


#DQB = [4,16]
DQB = [3,6]
#change_fixed_bitwdith(DQB[0])
change_msfp_bitwdith(DQB[0],8)
Dynamic_quantisation = 1
#run_NC_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss1)
#run_LP_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss4)
run_DQ_GC_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB)
#run_ppi_DQ_training_loop(accuracy_list_dq,Dynamic_quantisation, DQB, loss4)
print(accuracy_list_dq)

plt.plot(range(1, 151), loss1, color='red', label='ADQF = [2, 4]')
plt.plot(range(1, 151), loss2, color='blue',label='ADQF = [2, 6]')
plt.plot(range(1, 151), loss3, color='green',label='ADQF = [3, 5]')
plt.plot(range(1, 151), loss4, color='purple',label='ADQF = [3, 6]')
#plt.plot(range(1, 151), loss5, color='magenta',label='SDQF = [8, 4, 4, 8]')
plt.xlabel('Epoch', fontsize=30)
plt.ylabel('Loss', fontsize=30)
plt.title('Validation Loss vs Epochs using MSFP quantization with an active dynamic quantization scheme', fontsize=30)
plt.legend(fontsize=28)
plt.show()

























#####################################
#####################################
#####################################



#change_msfp_bitwdith(8, 8)
#reload_layers()
#run_NC_training_loop(accuracy_list_dq)
#print(accuracy_list_dq)


#change_to_normal()
#reload_layers()
#run_NC_training_loop(accuracy_list_dq)
#print(accuracy_list_dq)

#train_NC()

#""" def normal():
#    change_to_normal()what is
#    time.sleep(3)
#    run_NC_training_loop(accuracy_list)
#    print(accuracy_list)

#def fixed(bits):
#    change_fixed_bitwdith(bits)
#    subprocess.run(["python", r"C:\Users\IBqai\Desktop\GNN\code\src\Fixed_and_MSFP_quantize\uniform_quantize.py"])
#    run_NC_training_loop(accuracy_list_fixed)
#    print(accuracy_list_fixed)

#change_msfp_bitwdith(8,4)
#subprocess.run(["python", r"C:\Users\IBqai\Desktop\GNN\code\src\Fixed_and_MSFP_quantize\train_NC_gpu.py"])
#exec(open(r"C:\Users\IBqai\Desktop\GNN\code\src\Fixed_and_MSFP_quantize\train_NC_gpu.py").read())
#run_NC_training_loop(accuracy_list_fixed)
#print(accuracy_list_fixed) """








#""" for i in range(1,5):
#    change_fixed_bitwdith(i*2)
#    subprocess.run(["python", r"C:\Users\IBqai\Desktop\GNN\code\src\Fixed_and_MSFP_quantize\quantised_gcn_layer.py"])
#    run_NC_training_loop(accuracy_list)
#print(accuracy_list)"""



""" print('Starting base')
change_to_normal()
run_NC_training_loop(accuracy_list)
print(accuracy_list)

print('Starting fixed')
change_fixed_bitwdith(8)
run_NC_training_loop(accuracy_list_fixed)
change_fixed_bitwdith(6)
run_NC_training_loop(accuracy_list_fixed)
change_fixed_bitwdith(4)
run_NC_training_loop(accuracy_list_fixed)
change_fixed_bitwdith(2)
run_NC_training_loop(accuracy_list_fixed)
print(accuracy_list_fixed)

print('Starting msfp')
change_msfp_bitwdith(4,8)
run_NC_training_loop(accuracy_list_msfp)
change_msfp_bitwdith(5,8)
run_NC_training_loop(accuracy_list_msfp)
change_msfp_bitwdith(6,8)
run_NC_training_loop(accuracy_list_msfp)
change_msfp_bitwdith(7,8)
run_NC_training_loop(accuracy_list_msfp)
change_msfp_bitwdith(8,8)
run_NC_training_loop(accuracy_list_msfp)
print(accuracy_list)
print(accuracy_list_fixed)
print(accuracy_list_msfp)  """

#change_msfp_bitwdith(16,8)
#run_NC_training_loop(test)
#print(test)

""" print('starting normal')
change_to_normal()
run_NC_training_loop(test)
print(test)

print('Starting fixed')
change_fixed_bitwdith(8)
run_NC_training_loop(accuracy_list_fixed)
change_fixed_bitwdith(6)
run_NC_training_loop(accuracy_list_fixed)
change_fixed_bitwdith(4)
run_NC_training_loop(accuracy_list_fixed)
change_fixed_bitwdith(8)
run_NC_training_loop(accuracy_list_fixed)
print(accuracy_list_fixed) """