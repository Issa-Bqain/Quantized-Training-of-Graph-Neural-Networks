# Inherit from Function
from torch.autograd import Function
import torch
from torch import nn
from torch.nn import functional as F
from math import ceil, log2, sqrt
from torch.autograd.function import InplaceFunction

from math import ceil, log2
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor

from utils import block, my_clamp, my_round, unblock
import json
import os

#file_path = os.path.join(os.getcwd(), 'config.json')
file_path_json = r'C:\Users\IBqai\Desktop\GNN\code\src\Fixed_and_MSFP_quantize\bitconfig.json'

#with open(file_path, 'r') as f:
#    bitconfig = json.load(f)

def load_json_file(file_path):
    with open(file_path) as f:
        return json.load(f)

bitconfig = load_json_file(file_path_json)

def reload_json_data():
    global json_data
    json_data = load_json_file('your_file.json')

#init fixed
xbits = bitconfig["xbits"]
xpercentile = 0.99
#xbucket_size = 0

#init msfp
xmantissap1 = bitconfig["xmantissap1"]
xexponent_width = bitconfig["xexponent_width"]

#print("fixed, bits: " , xbits)
#print("msfp, mantissa: " , xmantissap1, "exponent", xexponent_width)



class my_clamp_func(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class my_round_func(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    


P2 = my_clamp_func.apply
P3 = my_round_func.apply

def get_dynamic_scale(x, bits, with_grad=False, percentile=1):
    """Calculate dynamic scale for quantization from input by taking the
    maximum absolute value from x and number of bits"""
    with torch.set_grad_enabled(with_grad):
        #import pdb;pdb.set_trace()
        rex = x.reshape(-1,)
        k = len(rex)
        # import pdb; pdb.set_trace()
        threshold = torch.topk(rex.abs(), k, largest=False)[0].max()
    return get_scale(bits, threshold)

def get_scale(bits, threshold):
    """Calculate scale for quantization according to some constant and number of bits"""
    try:
        return ceil(log2(threshold))
    except ValueError:
        return 0


def quantize(input, bits, percentile): # bits = 32
    """Do linear quantization to input according to a scale and number of bits"""
    thresh =  2 ** (bits - 1) - 1
    #scale = 2**24
    scale = 2 ** (bits - get_dynamic_scale(input, bits=bits, percentile=percentile) - 1)
    #import pdb; pdb.set_trace()
    return P2(P3(input.mul(scale)), -thresh, thresh).div(scale)


def msfloat_quantize_act(input, bits, percentile, bucket_size):
    # padd zeros so that the size of the input is a multiple of the bucket size
    if len(input.shape) == 2:
        batch, size = input.shape
    if len(input.shape) == 3:
        batch, pd, size = input.shape
        
    num_buckets = ceil(size / bucket_size)
    new_size = num_buckets * bucket_size
    diff = new_size - size
    padded = F.pad(input, (0, diff))
    # reshape to be a multiple of the bucket size, so that we can quantize it in chunks
    if len(input.shape) == 2:
        padded = padded.reshape(batch, num_buckets, bucket_size)     
    if len(input.shape) == 3:
        padded = padded.reshape(batch, pd, num_buckets, bucket_size)

    # probably we should do top_k here using percentile? 
    # but this should not matter for small bucket sizes
    
    if len(input.shape) == 2:
        pre_log_ceil = padded.abs().max(dim=2)[0]
    if len(input.shape) == 3:
        pre_log_ceil = padded.abs().max(dim=3)[0]
    pre_log_ceil[pre_log_ceil==0] = pre_log_ceil[pre_log_ceil!=0].min()

    # compute scale
    max_value =  2 ** (bits - 1) - 1
    scale = torch.ceil(torch.log2(pre_log_ceil))
    scale = 2 ** (bits - scale - 1)
    # import pdb; pdb.set_trace()
    # get quantized number through rounding and clamping
    if len(input.shape) == 2:
        scale = scale.unsqueeze(2)
    if len(input.shape) == 3:
        scale = scale.unsqueeze(3)

    quantized = P2(P3(padded.mul(scale)),-max_value, max_value).div(scale)
    if len(input.shape) == 2:
        quantized = quantized.reshape(batch, -1)[:, :size]
    if len(input.shape) == 3:
        quantized = quantized.reshape(batch, pd, -1)[:, :, :size]
    return quantized





def msfp_quantizer(
    x: torch.Tensor,
    width: int = 12,
    exponent_width: int = 8,
    exponent_bias: int = None,
    block_size: List[int] = [16],
    skip_first_dim: bool = True,
):
    """
    - Convert IEEE FP32/64 to Microsoft floating point (MSFP), where an exponent is shared over all elements in a block.
    - `e_shared x [(-1)^s1 x mantissa1, (-1)^s2 x mantissa2, ...]`
    - See https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf
    ---
    - forward: convert IEEE FP32/64 to MSFP
    - backward: STE
    ---
    - `width`: The number of mantissa bits + 1 (the sign bit)
    - `exponent_width`: the number of exponent bits, which is shared over a block
    - `exponent_bias`: the exponent bias, if None, `2**(exponent_bits-1)-1` will be used
    - `block_size`: a list of integers where each integer is the block size on that dimension. See function `block`.
    """
    if isinstance(block_size, int):
        block_size = [block_size]
    # separate x into blocks
    x_shape_before_blocking = [i for i in x.shape]
    blocked_x, per_block_max, padded_x_shape, block_shape = block(
        x, block_shape=block_size, skip_first_dim=skip_first_dim
    )
    # TODO: Why we have all zero bias
    # fill zeros to avoid log2(0) = -inf
    if torch.all(per_block_max == 0):
        per_block_max = torch.ones_like(per_block_max)
    else:
        per_block_max[per_block_max == 0] = per_block_max[per_block_max != 0].min()
    # minifloat_simple_quantizer on each block over which a exponent is shared
    mantissa_bits = width - 1
    if exponent_bias in (None, "none", "None"):
        exponent_bias = 2 ** (exponent_width - 1) - 1

    exponent_max = 2**exponent_width - 1 - exponent_bias
    exponent_min = -exponent_bias

    mantissa_integer_max = 2**mantissa_bits - 1
    # sign
    per_block_sign = torch.sign(blocked_x + 1e-9)
    # exponent
    per_block_value = torch.abs(blocked_x) + 1e-9
    per_block_exponent = torch.ceil(torch.log2(per_block_max))
    per_block_exponent = my_clamp(per_block_exponent, exponent_min, exponent_max)
    # mantissa
    per_block_mantissa = per_block_value / 2**per_block_exponent
    shift = 2**mantissa_bits
    per_block_mantissa_integer = my_clamp(
        my_round(per_block_mantissa * shift), 0, mantissa_integer_max
    )
    per_block_mantissa = per_block_mantissa_integer / shift

    per_block_msfp = per_block_sign * (2**per_block_exponent) * per_block_mantissa
    msfp_x = unblock(
        per_block_msfp,
        x_shape_before_blocking=x_shape_before_blocking,
        padded_x_shape=padded_x_shape,
        block_shape=block_shape,
        skipped_first_dim_when_blocking=skip_first_dim,
    )

    # fmt: off
    # this `is_close_to_0` helps the grad keeps 1 if input x is 0, or the zero-initialized value will be trapped in 0
    is_close_to_0 = torch.isclose(x, torch.tensor([0.0], dtype=x.dtype, device=x.device))
    msfp_x = (~is_close_to_0) * msfp_x + (is_close_to_0) * x
    # fmt: on
    return msfp_x






class FixedLinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, config=None):
        #import pdb; pdb.set_trace()
        ctx.save_for_backward(quantize(input, xbits, xpercentile),
                quantize(weight,  xbits, xpercentile),
                bias)
        ctx.config = config
        w_q = quantize(weight,  xbits, xpercentile)
        output = quantize(input,  xbits, xpercentile).matmul(w_q.t())
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        #optional inputs.
        
        #import pdb; pdb.set_trace()
       
        input, weight, bias = ctx.saved_tensors
        #import pdb; pdb.set_trace()
        q_args = ctx.config
        grad_input = grad_weight = grad_bias = None
        #import pdb; pdb.set_trace()


        grad_output = quantize(grad_output, xbits, xpercentile)

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        #import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        
        #grad_input = grad_output.matmul(weight)
        #grad_weight = grad_output.permute(0,2,1).matmul(input)

        grad_input = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.t(), input)
        
        if bias is not None:
            grad_bias = grad_output.sum(0)
        

        return grad_input, grad_weight, grad_bias, None, None

class MSFPLinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, config=None):
        #import pdb; pdb.set_trace()
        ctx.save_for_backward(msfp_quantizer(x=input, width =xmantissap1 , exponent_width=xexponent_width),
                msfp_quantizer(x=weight, width =xmantissap1 , exponent_width=xexponent_width),
                bias)
        ctx.config = config
        w_q = msfp_quantizer(x=weight, width =xmantissap1 , exponent_width=xexponent_width)
        output = msfp_quantizer(x=input, width =xmantissap1 , exponent_width=xexponent_width).matmul(w_q.t())
        #print('###################################################')
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        #optional inputs.
        
        #import pdb; pdb.set_trace()
       
        input, weight, bias = ctx.saved_tensors
        #import pdb; pdb.set_trace()
        q_args = ctx.config
        grad_input = grad_weight = grad_bias = None
        #import pdb; pdb.set_trace()


        grad_output = msfp_quantizer(x=grad_output, width =xmantissap1 , exponent_width=xexponent_width)

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        #import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        
        #grad_input = grad_output.matmul(weight)
        #grad_weight = grad_output.permute(0,2,1).matmul(input)

        grad_input = torch.matmul(grad_output, weight)
        grad_weight = torch.matmul(grad_output.t(), input)
        
        if bias is not None:
            grad_bias = grad_output.sum(0)
        

        return grad_input, grad_weight, grad_bias, None, None




""" class MSFPLinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, config=None):
        msq_input = msfp_quantizer(x=input, width =xmantissap1 , exponent_width=xexponent_width)
        msq_w = msfp_quantizer(x=input, width =xmantissap1 , exponent_width=xexponent_width)
        
        # using norms to check, the msfloat quant should give a smaller rounding error
        # print(torch.norm(msq_input - input))
        # print(torch.norm(q_input - input))
        # import pdb; pdb.set_trace()
        ctx.save_for_backward(msq_input, msq_w, bias)
        ctx.config = config
        w_q = msfp_quantizer(x=input, width =xmantissap1 , exponent_width=xexponent_width)
        output = msfp_quantizer(x=input, width =xmantissap1 , exponent_width=xexponent_width).matmul(w_q.t())
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        q_args = ctx.config
        grad_input = grad_weight = grad_bias = None
        #import pdb; pdb.set_trace()

        #if xpercentile != 0:
        grad_output = msfp_quantizer(x=input, width =xmantissap1 , exponent_width=xexponent_width)


        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        #import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        grad_input = grad_output.matmul(weight)
        #import pdb; pdb.set_trace()
        if len(grad_output.shape) == 2:
            grad_weight = grad_output.permute(1,0).matmul(input)
        if len(grad_output.shape) == 3:
            grad_weight = grad_output.permute(0,2,1).matmul(input)

        if bias is not None:
            grad_bias = grad_output.sum(0)
        #import pdb; pdb.set_trace()

        return grad_input, grad_weight, grad_bias, None, None """
    
    
    

class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, config=None):
        #import pdb; pdb.set_trace()
        ctx.save_for_backward(input, weight, bias)
        ctx.config = config
        output = input.matmul(weight.t())
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        #import pdb; pdb.set_trace()
        q_args = ctx.config
        grad_input = grad_weight = grad_bias = None
        
        grad_input = grad_output.matmul(weight)
        grad_weight = grad_output.permute(0,2,1).matmul(input)
        
        if bias is not None:
            grad_bias = grad_output.sum(0)
        #import pdb; pdb.set_trace()

        return grad_input, grad_weight, grad_bias, None, None
    
    
class QLinear(nn.Module):

    def __init__(self, input_features, output_features, config, bias=True):
        super(QLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.config = config
        
        #if self.config['dynamicstashing']:
        #    loss_avg = []
        #    setattr(self, 'loss_avg', loss_avg)
        
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        #nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        #if self.bias is not None:
            #nn.init.uniform_(self.bias, -0.1, 0.1)
        #import pdb; pdb.set_trace()
        # force to set attr in Func
        if self.config['qscheme'] == 'fixed':
            FixedLinearFunction.config = self.config
        if self.config['qscheme'] == 'msfp':
            MSFPLinearFunction.config = self.config 

   
    def forward(self, input):
        # import pdb; pdb.set_trace()
        # See the autograd section for explanation of what happens here.
        
        if not hasattr(self, 'loss'):
            self.loss = 'n'
        
        if self.config['qscheme'] == 'fixed':
            return FixedLinearFunction.apply(input, self.weight, self.bias, self.config) #, self.loss)
        if self.config['qscheme'] == 'msfp':
            return MSFPLinearFunction.apply(input, self.weight, self.bias, self.config) #,, self.loss)
        if self.config['qscheme'] == 'linear':
            return LinearFunction.apply(input, self.weight, self.bias, self.config) #,, self.loss)
        
#tensor_random = torch.randn((5, 5))
#print(tensor_random)
#quant = msfp_quantizer(x=tensor_random, width =xmantissap1 , exponent_width=xexponent_width)

#print(quant)