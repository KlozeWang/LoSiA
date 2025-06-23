from torch import nn
import torch
from torch.autograd import Function
import math
from torch.nn import functional as F, init

class SelectiveLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, keys_x=None, keys_y=None):
        ctx.keys_x = keys_x
        ctx.keys_y = keys_y
        ctx.save_for_backward(
            input if keys_y is None else input[:,:, keys_y],
            weight,
            bias if bias is not None else None
        )
        
        with torch.no_grad():
            output = input @ weight.t()
            if bias is not None:
                output += bias
            return output

    @staticmethod
    def backward(ctx, grad_output):
        input_reduced, weight, bias = ctx.saved_tensors
        keys_x = ctx.keys_x
        keys_y = ctx.keys_y
        
        grad_input = grad_weight = grad_bias = None
        grad_input = grad_output.matmul(weight)
        
        grad_weight = torch.zeros_like(weight)
        input_flat = input_reduced.reshape(-1, input_reduced.size(-1))
        
        if keys_y is not None:
            grad_flat = grad_output.reshape(-1, grad_output.size(-1))[:,keys_x]
            grad_weight[torch.meshgrid(keys_x, keys_y, indexing='ij')] = grad_flat.t() @ input_flat

        else:
            grad_flat = grad_output.reshape(-1, grad_output.size(-1))
            grad_weight = grad_flat.t() @ input_flat
        
        if bias is not None:
            grad_bias = torch.zeros_like(bias)
            grad_flat = grad_output.reshape(-1, grad_output.size(-1))
            
            if keys_x is not None:
                grad_bias[keys_x] = grad_flat[:, keys_x].sum(1)
            else:
                grad_bias = grad_flat.sum(1)
        
        return grad_input, grad_weight, grad_bias, None, None

class SelectiveLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SelectiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.keys_x = None
        self.keys_y = None
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
    def setxy(self, keys_x, keys_y):
        self.keys_x = keys_x
        self.keys_y = keys_y
    
    def forward(self, input):
        return SelectiveLinearFunction.apply(input, self.weight, self.bias, self.keys_x, self.keys_y)

def replace_linear_recursive(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_layer = SelectiveLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None
            )
            
            new_layer.weight = child.weight
            if child.bias is not None:
                new_layer.bias = child.bias
            new_layer.keys_x = torch.Tensor([0]).to(new_layer.weight.device, dtype=torch.int)
            new_layer.keys_y = torch.Tensor([0]).to(new_layer.weight.device, dtype=torch.int)
            
            setattr(module, name, new_layer)
        else:
            replace_linear_recursive(child)


def find_module_for_parameter(model, param):
    for name, module in model.named_modules():
        for p_name, p in module.named_parameters(recurse=False):
            if p is param:
                return module
    return None