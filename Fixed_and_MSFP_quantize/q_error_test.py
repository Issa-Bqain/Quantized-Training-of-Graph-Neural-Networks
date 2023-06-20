import torch
from uniform_quantize import FixedLinearFunction, MSFPLinearFunction
from uniform_quantize import msfp_quantizer, quantize

#xmantissap1 = 4
#xexponent_width = 8
#random_tensor = torch.randn(5, 5)

#print(random_tensor)
#random_tensor_msfp_small = msfp_quantizer(x=random_tensor, width =xmantissap1 , exponent_width=xexponent_width)
#print(random_tensor_msfp_small)

#random_tensor_msfp_big = msfp_quantizer(x=random_tensor, width =(xmantissap1+10) , exponent_width=xexponent_width)
#print(random_tensor_msfp_big)

#for x in range(1,101):
    #torch.manual_seed(1234567)
""" randomsmsfp = torch.rand((100, 100))
bitwidths = [2, 4, 8, 12, 16, 20, 24, 28, 32]
quantizedmsfp = [msfp_quantizer(x=randomsmsfp, width=bitwidth, block_size=16) for bitwidth in bitwidths]
roundoff_errors_msfp = [torch.sum(torch.abs(randomsmsfp - quantizedmsfp[i])) for i in range(len(quantizedmsfp))]
print(roundoff_errors_msfp)

#torch.manual_seed(1234567)
randoms = torch.rand((100, 100))
bitwidths = [2, 4, 8, 12, 16, 20, 24, 28, 32]
quantized = [quantize(input=randoms, bits=bitwidth, percentile=0.99) for bitwidth in bitwidths]
roundoff_errors_fixed = [torch.sum(torch.abs(randoms - quantized[i])) for i in range(len(quantized))]
print(roundoff_errors_fixed)

print("##################################################################")
print("##################################################################")
print("##################################################################")
print(roundoff_errors_msfp)
print(roundoff_errors_fixed) """


l0norm = 0
l1norm = 0
l2norm = 0

l0normfixed = 0
l1normfixed = 0
l2normfixed = 0

l0normmsfp = 0
l1normmsfp = 0
l2normmsfp = 0


for i in range(1,1000):
    random = torch.rand((100, 100))
    
    bitwidthmsfp = 8
    bitwidthfixed = 8
    randommsfp = msfp_quantizer(x=random, width=bitwidthmsfp, block_size=16)
    randomfixed = quantize(input=random, bits=bitwidthfixed, percentile=0.99)


    l0norm = l0norm + random.norm(p=0)
    l1norm = l1norm + random.norm(p=1)
    l2norm = l2norm + random.norm(p=2)


    l0normfixed = l0normfixed + randomfixed.norm(p=0)
    l1normfixed = l1normfixed + randomfixed.norm(p=1)
    l2normfixed = l2normfixed + randomfixed.norm(p=2)

    l0normmsfp = l0normmsfp + randommsfp.norm(p=0)
    l1normmsfp = l1normmsfp + randommsfp.norm(p=1)
    l2normmsfp = l2normmsfp + randommsfp.norm(p=2)



l0norm = l0norm / 1000
l1norm = l1norm / 1000
l2norm = l2norm / 1000

l0normfixed = l0normfixed /1000
l1normfixed = l1normfixed /1000
l2normfixed = l2normfixed /1000

l0normmsfp = l0normmsfp / 1000
l1normmsfp = l1normmsfp / 1000
l2normmsfp = l2normmsfp / 1000

print("L0 norm:", l0norm)
print("L1 norm:", l1norm)
print("L2 norm:", l2norm)

print("L0 normfixed:", l0normfixed)
print("L1 normfixed:", l1normfixed)
print("L2 normfixed:", l2normfixed)

print("L0 normmsfp:", l0normmsfp)
print("L1 normmsfp:", l1normmsfp)
print("L2 normmsfp:", l2normmsfp)






#randoms = torch.rand((100, 100))
#for i in bitwidths:
#    qrandoms = msfp_quantizer(randoms, width=i, block_size=16)
#    error = torch.abs(randoms - qrandoms)  # Calculate element-wise absolute difference
#    error = error.sum()       # Sum all the elements to get the total error
#    return error.item()       # Convert error tensor to a Python scalar

#    # Example usage
#    A = torch.randn(10, 10)    # Random tensor A
#    B = torch.round(A)         # Quantized tensor B

#error = quantization_error(A, B)




#model.eval()
# Generate a random input graph.
#batch_size = 2
#x, edge_index = data.x, data.edge_index
#x, edge_index = x.to(device), edge_index.to(device)

# Check the dimensions of the input data.
#print(f"Input data shape: {x.shape}, {edge_index.shape}")

# Compute the FLOPS count of the model.
#flops, params = profile(model, inputs=(x, edge_index))
#print(f"FLOPS: {flops:.2f}")


#flops, params = pf(model, inputs=(data.x.to(device),data.edge_index.to(device) ))
#print(flops)
#flops = FlopCountAnalysis(model, input)
#print(flops.total())
#print(flops.by_module())

#with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available(), profile_memory=True, record_shapes=True, with_flops=True) as prof:

#torch.cuda.synchronize() # wait for warm-up to finish
#times = []