from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np 
from collections import OrderedDict


class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_lv):    
        return input.mul(n_lv-1).round_().div_(n_lv-1)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Q_ReLU(nn.Module):
    def __init__(self, act_func=True, inplace=False):
        super(Q_ReLU, self).__init__()
        self.n_lv = 0
        self.act_func = act_func
        self.inplace = inplace
        self.a = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))

    def initialize(self, n_lv, offset, diff):
        self.n_lv = n_lv
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
    
    def forward(self, x):
        if self.act_func:
            x = F.relu(x, self.inplace)

        if self.n_lv == 0:
            return x
        else:
            a = F.softplus(self.a)
            c = F.softplus(self.c)
            x = F.hardtanh(x / a, 0, 1)
            x = RoundQuant.apply(x, self.n_lv) * c
            return x 

        
class Q_ReLU6(Q_ReLU):
    def __init__(self, act_func=True, inplace=False):
        super(Q_ReLU6, self).__init__(act_func, inplace)

    def initialize(self, n_lv, offset, diff):
        self.n_lv = n_lv
        if offset + diff > 6:
            self.a.data.fill_(np.log(np.exp(6)-1))
            self.c.data.fill_(np.log(np.exp(6)-1))
        else:
            self.a.data.fill_(np.log(np.exp(offset + diff)-1))
            self.c.data.fill_(np.log(np.exp(offset + diff)-1))


class Q_Sym(nn.Module):
    def __init__(self):
        super(Q_Sym, self).__init__()
        self.n_lv = 0
        self.a = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))

    def initialize(self, n_lv, offset, diff):
        self.n_lv = n_lv
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
    
    def forward(self, x):
        if self.n_lv == 0:
            return x
        else:
            a = F.softplus(self.a)
            c = F.softplus(self.c)

            x = F.hardtanh(x / a, -1, 1)
            x = RoundQuant.apply(x, self.n_lv // 2) * c
            return x 


class Q_HSwish(nn.Module):
    def __init__(self, act_func=True):
        super(Q_HSwish, self).__init__()
        self.n_lv = 0
        self.act_func = act_func
        self.a = Parameter(torch.Tensor(1))
        self.b = 3/8
        self.c = Parameter(torch.Tensor(1))
        self.d = -3/8

    def initialize(self, n_lv, offset, diff):
        self.n_lv = n_lv
        self.a.data.fill_(np.log(np.exp(offset + diff)-1))
        self.c.data.fill_(np.log(np.exp(offset + diff)-1))
    
    def forward(self, x):
        if self.act_func:
            x = x * (F.hardtanh(x + 3, 0, 6) / 6)

        if self.n_lv == 0:
            return x
        else:
            a = F.softplus(self.a)
            c = F.softplus(self.c)
            x = x + self.b
            x = F.hardtanh(x / a, 0, 1)
            x = RoundQuant.apply(x, self.n_lv) * c
            x = x + self.d
            return x 


class Q_Conv2d(nn.Conv2d):
    def __init__(self, *args, **kargs):
        super(Q_Conv2d, self).__init__(*args, **kargs)
        self.n_lv = 0
        self.a = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))
        self.weight_old = None

    def initialize(self, n_lv):
        self.n_lv = n_lv
        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))

    def _weight_quant(self):
        a = F.softplus(self.a)
        c = F.softplus(self.c)

        weight = F.hardtanh(self.weight / a, -1, 1)
        weight = RoundQuant.apply(weight, self.n_lv // 2) * c
        return weight

    def forward(self, x):
        if self.n_lv == 0:
            return F.conv2d(x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups)
        else:
            weight = self._weight_quant()

            return F.conv2d(x, weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups)


class Q_Linear(nn.Linear):
    def __init__(self, *args, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.n_lv = 0
        self.a = Parameter(torch.Tensor(1))
        self.c = Parameter(torch.Tensor(1))
        self.weight_old = None

    def initialize(self, n_lv):
        self.n_lv = n_lv
        max_val = self.weight.data.abs().max().item()
        self.a.data.fill_(np.log(np.exp(max_val * 0.9)-1))
        self.c.data.fill_(np.log(np.exp(max_val * 0.9)-1))

    def _weight_quant(self):
        a = F.softplus(self.a)
        c = F.softplus(self.c)

        weight = F.hardtanh(self.weight / a, -1, 1)
        weight = RoundQuant.apply(weight, self.n_lv // 2)  * c
        return weight

    def forward(self, x):
        if self.n_lv == 0:
            return F.linear(x, self.weight, self.bias)
        else:
            weight = self._weight_quant()
            return F.linear(x, weight, self.bias)            


class Q_Conv2dPad(Q_Conv2d):
    def __init__(self, mode, *args, **kargs):
        super(Q_Conv2dPad, self).__init__(*args, **kargs)
        self.mode = mode

    def forward(self, inputs):
        if self.mode == "HS":
            inputs = F.pad(inputs, self.padding + self.padding, value=-3/8)
        elif self.mode == "RE":
            inputs = F.pad(inputs, self.padding + self.padding, value=0)
        else:
            raise LookupError("Unknown nonlinear")

        if self.n_lv == 0:
            return F.conv2d(inputs, self.weight, self.bias,
                self.stride, 0, self.dilation, self.groups)
        else:
            weight = self._weight_quant()

            return F.conv2d(inputs, weight, self.bias,
                self.stride, 0, self.dilation, self.groups)



def initialize(model, loader, n_lv, act=False, weight=False, eps=0.05):
    def initialize_hook(module, input, output):
        if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)) and act:
            if not isinstance(input, torch.Tensor):
                input = input[0]
            input = input.detach().cpu().numpy()

            if isinstance(input, Q_Sym):
                input = np.abs(input)
            elif isinstance(input, Q_HSwish):
                input = input + 3/8

            input = input.reshape(-1)
            input = input[input > 0]
            input = np.sort(input)
            
            if len(input) == 0:
                small, large = 0, 1e-3
            else:
                small, large = input[int(len(input) * eps)], input[int(len(input) * (1-eps))]

            module.initialize(n_lv, small, large - small)

        if isinstance(module, (Q_Conv2d, Q_Linear)) and weight:
            module.initialize(n_lv)

    hooks = []

    for name, module in model.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)

    
    model.train()
    model.cpu()
    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                output = model.module(input)
            else:
                output = model(input)
        break
    
    model.cuda()
    for hook in hooks:
        hook.remove()


class Q_Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Q_Sequential, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            idx = 0 
            for module in args:
                if isinstance(module, Q_Sym) or (isinstance(module, Q_HSwish) and idx == 0):
                    self.add_module('-' + str(idx), module)
                else:
                    self.add_module(str(idx), module)
                    idx += 1


class QuantOps(object):
    initialize = initialize
    Conv2d = Q_Conv2d
    ReLU = Q_ReLU
    ReLU6 = Q_ReLU6
    Sym = Q_Sym
    HSwish = Q_HSwish
    Conv2dPad = Q_Conv2dPad        
    Sequential = Q_Sequential
    Linear = Q_Linear