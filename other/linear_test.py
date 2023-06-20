import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, qbits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.qbits = qbits
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
        
        # quantization parameters for input and weight
        self.input_scale = nn.Parameter(torch.Tensor([1.0]))
        self.input_zero_point = nn.Parameter(torch.Tensor([0]).to(torch.int32))
        self.weight_scale = nn.Parameter(torch.Tensor([1.0]))
        self.weight_zero_point = nn.Parameter(torch.Tensor([0]).to(torch.int32))
        
    def forward(self, x):
        # quantize input tensor
        x = torch.quantize_per_tensor(x, scale=self.input_scale, zero_point=self.input_zero_point, dtype=torch.quint8)
        x = x.dequantize()
        
        # quantize weight tensor
        weight_q = torch.quantize_per_tensor(self.weight, scale=self.weight_scale, zero_point=self.weight_zero_point, dtype=torch.qint8)
        weight_q = weight_q.dequantize()
        
        # perform quantized linear transformation
        output = F.linear(x, weight_q, self.bias)
        
        # quantize output tensor
        output = torch.quantize_per_tensor(output, scale=self.output_scale, zero_point=self.output_zero_point, dtype=torch.quint8)
        
        return output
    
    def backward(self, grad_output):
        # quantize gradient tensor
        grad_output_q = torch.quantize_per_tensor(grad_output, scale=self.output_scale, zero_point=self.output_zero_point, dtype=torch.qint8)
        grad_output_q = grad_output_q.dequantize()
        
        # quantize weight tensor
        weight_q = torch.quantize_per_tensor(self.weight, scale=self.weight_scale, zero_point=self.weight_zero_point, dtype=torch.qint8)
        weight_q = weight_q.dequantize()
        
        # compute gradients w.r.t. weight and bias parameters
        grad_weight = F.linear(grad_output_q, torch.quantize_per_tensor(self.input, scale=self.input_scale, zero_point=self.input_zero_point, dtype=torch.qint8), None)
        grad_bias = None
        if self.bias is not None:
            grad_bias = grad_output_q.sum(dim=0)
        
        # quantize gradients
        grad_weight = torch.quantize_per_tensor(grad_weight.dequantize(), scale=self.weight_scale, zero_point=self.weight_zero_point, dtype=torch.qint8)
        if grad_bias is not None:
            grad_bias = torch.quantize_per_tensor(grad_bias.dequantize(), scale=self.weight_scale, zero_point=self.weight_zero_point, dtype=torch.qint8)
        
        return grad_weight, grad_bias
    
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, qbits={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.qbits
        )
