import math
from functools import partial

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn



# #svd分解的lora
# def reset_parameters(self):
#     nn.Linear.reset_parameters(self) # note: we can try fixed initalized parameters
#     if hasattr(self, 'lora_A'):
#         # initialize A the same way as the default for nn.Linear and B to zero
#         u, s, v = torch.linalg.svd(self.original_weights)
#         self.lora_A.data = (u[:, :self.rank] * s[:self.rank]).T
#         self.lora_B.data = v[:, :self.rank] 
#         self.prev_A.data.copy_(self.lora_A.data)  # 初始化prev_A
#         self.prev_B.data.copy_(self.lora_B.data) 

class SaRAParametrization(nn.Module):
    def __init__(self, fan_in, fan_out, fan_in_fan_out=False, rank=4, lora_dropout_p=0.0, lora_alpha=8):
        super().__init__() 
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        self.lora_A = nn.Parameter(torch.zeros(self.swap((rank, fan_in))))
        self.lora_B = nn.Parameter(torch.zeros(self.swap((fan_out, rank))))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.lora_dropout = nn.Dropout(p=lora_dropout_p) if lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones(self.swap((1, fan_in)), dtype=self.lora_A.dtype))
        self.forward_fn = self.lora_forward

    def _dropout(self, A):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)

    def sara_forward(self, X):
        return X + torch.matmul(*self.swap((self.lora_B, self.dropout_fn(self.lora_A)))).view(X.shape) * self.scaling

    def forward(self, X):
        return self.forward_fn(X)

    def disable_sara(self):
        self.forward_fn = lambda x: x

    def enable_sara(self):
        self.forward_fn = self.sara_forward

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )

    @classmethod
    def from_conv2d(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )

    @classmethod
    def from_embedding(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=True, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha
        )


default_sara_config = {  # specify which layers to add sara to, by default only add to linear layers
    nn.Linear: {
        "weight": partial(SaRAParametrization.from_linear, rank=8),
    },
}



def apply_sara(layer, register=True, merge=False, sara_config=default_sara_config):
    #    这行定义了一个函数`apply_sara`，它接受一个网络层（`layer`），三个可选参数`register`（默认为True），
    # `merge`（默认为False），和`sara_config`（默认为`default_sara_config`）。

    """add sara parametrization to a layer, designed to be used with model.apply"""
    #    这是一个文档字符串，解释了这个函数的用途：给一个层添加SaRA参数化，通常用于与`model.apply`一起使用。

    if register:#    这个条件判断是检查是否需要注册SaRA参数化。
        
        if type(layer) in sara_config:#    如果当前层的类型在`sara_config`中定义了相应的SaRA参数化设置，则继续执行。
            
            for attr_name, parametrization in sara_config[type(layer)].items():
                #    遍历该层类型在`sara_config`中定义的所有SaRA参数化。

                parametrize.register_parametrization(layer, attr_name, parametrization(layer))
                #    对每个属性和参数化，调用`register_parametrization`来在层上注册SaRA参数化。

                
    else:  # this will remove all parametrizations, use with caution
    #    如果`register`为False，则进入这个分支，这个分支将移除所有参数化。

        if hasattr(layer, "parametrizations"):#    检查层是否有`parametrizations`属性。

            for attr_name in layer.parametrizations.keys():#    如果有，遍历所有的参数化属性。
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)
            #     移除每个参数化，如果`merge`为True，则在移除时保留参数化的结果。


#    定义了一个函数`add_sara`，接受一个模型和一个可选的SaRA配置（默认为`default_sara_config`）。
def add_sara(model, sara_config=default_sara_config):
    """add sara parametrization to all layers in a model. Calling it twice will add sara twice"""
    #    文档字符串说明了这个函数的功能：给模型中所有层添加SaRA参数化。如果调用两次，会添加两次SaRA。
    model.apply(partial(apply_sara, sara_config=sara_config))
#    使用`apply`方法在模型的所有层上应用`apply_sara`函数。这里用到了`partial`，
# 它创建了一个新的函数，将`sara_config`作为参数预先填充到`apply_sara`中。


def add_sara_by_name(model, target_module_names, sara_config=default_sara_config):
    """Add SaRA parameterization to specific layers in a model by names"""
    for name, layer in model.named_modules():
        if any([m in name for m in target_module_names]):
            add_sara(layer, sara_config=sara_config)


def merge_sara(model):
    """merge sara parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_sara, register=False, merge=True))


def remove_sara(model):
    """remove sara parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_sara, register=False, merge=False))




