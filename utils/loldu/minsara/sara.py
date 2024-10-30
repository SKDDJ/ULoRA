import math
from functools import partial
from typing import Any, Optional, Union
import torch
from torch import svd_lowrank
import torch.nn.utils.parametrize as parametrize
from torch import nn
import lightning as pl

import torch.nn.functional as F


def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, torch.nn.Parameter):
        return torch.nn.Parameter(weight.T)
    return weight.T


def ldu_decomposition(A):
    # Perform LU decomposition
    #todo: test no permutation matrix
    # P, L, U = torch.linalg.lu(A,pivot=False)
    P, L, U = torch.linalg.lu(A)
    # Extract the diagonal elements of U to form D
    D = torch.diagonal(U)
    # Normalize U to get the unit diagonal upper triangular matrix
    U = U / D.unsqueeze(-1)
    return P, L, D, U

class SaRAParametrization(pl.LightningModule):
    def __init__(self, fan_in, fan_out, fan_in_fan_out=False, rank=4, lora_dropout_p=0.0, lora_alpha=8, layer_module=None, init_sara_weights="ldu", init_method="lu" ):
        super().__init__() 
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        # 假设layer_module是一个已存在的层，我们用它的权重尺寸初始化layer_weights
        layer_weights = layer_module.weight
        
        self.layer_weights = layer_weights
        del layer_weights
        
        self.layer_weights.data = layer_module.weight.data

        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.init_method = init_method
        self.lora_dropout_p = lora_dropout_p

        # 注册一个前向传播前的钩子，以自动更新权重一次
        self._forward_pre_hook_handle = self.register_forward_pre_hook(self._update_weights_once)
        # 添加一个额外的属性来标记是否更新过权重
        self._updated = False
        # lu init
        self.init_sara_weights = init_sara_weights

        # todo: test removing the code below
        self.forward_fn = self.sara_forward
        

    def _update_weights_once(self, *args):
        if not self._updated:  # 只有在未更新权重的情况下才执行
            # initialization of A and B
            # note: because we may use weight tying, so we have to define the lora_X as nn.Parameter not the nn.Linear
            # , device=self.lora_A.device
            # , device=self.lora_A.device
            self.lora_A = nn.Parameter(torch.zeros(self.swap((self.rank, self.fan_in)))) # U
            self.lora_B = nn.Parameter(torch.zeros(self.swap((self.fan_out, self.rank)))) # L
            self.P = nn.Parameter(torch.zeros(self.swap((self.fan_out, self.fan_out)))) # P
            self.vector_z = nn.Parameter(torch.ones(self.rank, device=self.layer_weights.device)) 
            self.scaling_factor = nn.Parameter(torch.tensor(self.scaling, device=self.layer_weights.device))
            self.get_residual_matrix()
            self._forward_pre_hook_handle.remove()  # 移除钩子
            self._updated = True  # 更新标记，防止再次执
        # 权重更新后立即移除钩子，确保只执行一次
    # @torch.no_grad()  
    def get_residual_matrix(self):
        r = self.rank
        init_sara_weights = self.init_sara_weights
        
        weight = self.layer_weights 
        dtype = weight.dtype
        
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)

        
        # todo check if it is need to del teh U V S
        if init_sara_weights == "ldu":
            P, L, D, U = ldu_decomposition(weight.data)
            Lr = L[:, :r]
            Dr = D[:r]
            Ur = U[: r]
               
        lora_A = Ur
        lora_B = Lr
        vector_z = Dr
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B
        self.P.data = P
        
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False
        self.P.requires_grad = False
        
        init_method = self.init_method  # 可选项: 'lu', 'kaiming_normal', 'kaiming_uniform', 'uniform', 'normal', 'constant', 'ones', 'zeros'
        if init_method == 'lu':
            self.vector_z.data = vector_z
        elif init_method == 'suniform':
            nn.init.uniform_(self.vector_z, a=-1, b=1)
        elif init_method == 'uniform':
            mean_value = self.vector_z.mean().item()
            print(f"self.vector_z mean is {mean_value}")
            nn.init.uniform_(self.vector_z, a=-mean_value/2, b=mean_value/2)
        elif init_method == 'normal':
            mean_value = self.vector_z.mean().item()
            std_value = self.vector_z.std().item()
            print(f"self.vector_z mean is {mean_value}")
            print(f"self.vector_z std is {std_value}")
            nn.init.normal_(self.vector_z, mean=mean_value, std=std_value)
            # nn.init.normal_(self.vector_z, mean=0.0, std=1.0)
        elif init_method == 'constant':
            mean_value = self.vector_z.mean().item()
            print(f"self.vector_z mean is {mean_value}")
            nn.init.constant_(self.vector_z, mean_value)
            # nn.init.constant_(self.vector_z, 3.14)
        elif init_method == 'ones':
            nn.init.ones_(self.vector_z)
        elif init_method == 'zeros':
            nn.init.zeros_(self.vector_z)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
        
        
        # drop out which won't be used 
        self.lora_dropout = nn.Dropout(p=self.lora_dropout_p) if self.lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if self.lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones(self.swap((1, self.fan_in)), dtype=self.lora_A.dtype))
        
        resmat = self.layer_weights - self.scaling_factor * P @ lora_B @ torch.diag(vector_z) @lora_A         
        resmat = resmat.to(dtype)
        
        self.layer_weights.data = resmat
        del  P,resmat, lora_A, lora_B, vector_z, weight
        
    def _dropout(self, A):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)
  
    def sara_forward(self, X):
        torch_X_dtype = X.dtype #16
        diag_z = torch.diag(self.vector_z) # 32
        
        result = self.scaling_factor * self.P @ self.lora_B @ diag_z @ self.lora_A        
        result = result.to(torch_X_dtype) # 32 -> 16
        del diag_z
        # omit the original weights only return the LDU matrix
        return X + result
    
    def forward(self, X):
        return self.forward_fn(X)

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1, init_sara_weights="ldu", init_method="lu"):
        fan_out, fan_in = layer.weight.shape        
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha, layer_module=layer, init_sara_weights="ldu", init_method=init_method 
        )

    @classmethod
    def from_conv2d(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1, init_sara_weights="ldu", init_method="lu"):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        # layer_weights = layer
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha, layer_module=layer
        )

    @classmethod
    def from_embedding(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1, init_sara_weights="ldu", init_method="lu"):
        fan_in, fan_out = layer.weight.shape
        return cls(
            fan_in, fan_out, fan_in_fan_out=True, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha, layer_module=layer
        )


    def disable_sara(self):
        self.forward_fn = lambda x: x

    def enable_sara(self):
        self.forward_fn = self.sara_forward

default_sara_config = {  # specify which layers to add sara to, by default only add to linear layers
    nn.Linear: {
        "weight": partial(SaRAParametrization.from_linear, rank=768),
    },
}

def apply_sara(layer, register=True, merge=False, sara_config=default_sara_config):
    #    这行定义了一个函数`apply_sara`，它接受一个层（`layer`），三个可选参数`register`（默认为True），
    # `merge`（默认为False），和`sara_config`（默认为`default_sara_config`）。
    # merge_sara : register=False, merge=True
    """add sara parametrization to a layer, designed to be used with model.apply"""
    if register:#    这个条件判断是检查是否需要注册SaRA参数化。
        if type(layer) in sara_config:#    如果当前层的类型在`sara_config`中定义了相应的SaRA参数化设置，则继续执行。
            # print(sara_config[type(layer)])#    {'weight': functools.partial(<class '__main__.SaRAParametrization'>, rank=8)}
            for attr_name, parametrization in sara_config[type(layer)].items():
                # attr_name:"weight"; parametrization: partial(SaRAParametrization.from_linear, rank=8) 
                    parametrize.register_parametrization(layer, attr_name, parametrization(layer))
                
    else:  # this will remove all parametrizations, use with caution
    #    如果`register`为False，则进入这个分支，这个分支将移除所有参数化。
        if hasattr(layer, "parametrizations"):#    检查层是否有`parametrizations`属性。
            for attr_name in layer.parametrizations.keys():#    如果有，遍历所有的参数化属性。
                parametrize.remove_parametrizations(layer, attr_name, leave_parametrized=merge)
            #     移除每个参数化，如果`merge`为True，则在移除时保留参数化的结果。


#    定义了一个函数`add_sara`，接受一个模型和一个可选的SaRA配置（默认为`default_sara_config`）。
def add_sara(model, sara_config=default_sara_config):
    """add sara parametrization to all layers in a model. Calling it twice will add sara twice"""
    #给模型中所有层添加SaRA参数化。如果调用两次，会添加两次SaRA。
    model.apply(partial(apply_sara, sara_config=sara_config))
#    使用`apply`方法在模型的所有层上应用`apply_sara`函数。这里用到了`partial`，
# 它创建了一个新的函数，将`sara_config`作为参数预先填充到`apply_sara`中。


def add_sara_by_name(model, target_module_names, sara_config=default_sara_config):
    for name, layer in model.named_modules():
        # name: 0,1,2,...
        # layer: Sequential(...
        # for example if target_module_names = ['fc', 'conv'], then all layers whose name contains 'fc' or 'conv' will be added sara
        if any([m in name for m in target_module_names]):
            add_sara(layer, sara_config=sara_config)


def merge_sara(model):
    """merge sara parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_sara, register=False, merge=True))


def remove_sara(model):
    """remove sara parametrization to all layers in a model. This will remove all parametrization"""
    model.apply(partial(apply_sara, register=False, merge=False))

