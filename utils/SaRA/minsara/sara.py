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


class SaRAParametrization(pl.LightningModule):
    def __init__(self, fan_in, fan_out, fan_in_fan_out=False, rank=4, lora_dropout_p=0.0, lora_alpha=8, layer_module=None, init_sara_weights="fast_niter_8"):
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
        self.lora_dropout_p = lora_dropout_p

        # 注册一个前向传播前的钩子，以自动更新权重一次
        self._forward_pre_hook_handle = self.register_forward_pre_hook(self._update_weights_once)
        # 添加一个额外的属性来标记是否更新过权重
        self._updated = False
        # svd init
        self.init_sara_weights = init_sara_weights

        # todo: test removing the code below
        self.forward_fn = self.sara_forward
        

    def _update_weights_once(self, *args):
        if not self._updated:  # 只有在未更新权重的情况下才执行
            # initialization of A and B
            # note: because we may use weight tying, so we have to define the lora_X as nn.Parameter not the nn.Linear
            self.lora_A = nn.Parameter(torch.zeros(self.swap((self.rank, self.fan_in))))
            self.lora_B = nn.Parameter(torch.zeros(self.swap((self.fan_out, self.rank))))
            self.vector_z = nn.Parameter(self.rank * torch.ones(1))
            #TODO: TEST init a nn.Parameter name scaling_factor that is a scalar
            self.scaling_factor = nn.Parameter(torch.tensor(self.scaling))
            
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
        # print("weight dtype", dtype)
        
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        # print("new weight dtype", weight.dtype)

        
        # todo check if it is need to del teh U V S
        if init_sara_weights == "svd":
                V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
                Vr = V[:, : r]
                Sr = S[: r]
                Sr /= self.scaling
                Uhr = Uh[: r]
        elif len(init_sara_weights.split("_niter_")) == 2:           
            
                    # todo: note llama needs fp32 self.layer_weights.to(torch.float32)
                    Vr, Sr, Ur = svd_lowrank(
                        weight.data, r, niter=int(init_sara_weights.split("_niter_")[-1])
                    )
                    Sr /= self.scaling
                    Uhr = Ur.t()
            
        lora_A = Uhr
        lora_B = Vr
        vector_z = Sr
        # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_A.data = lora_A
        self.lora_B.data = lora_B
        self.vector_z.data = vector_z   
        
        # drop out which won't be used 
        self.lora_dropout = nn.Dropout(p=self.lora_dropout_p) if self.lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if self.lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones(self.swap((1, self.fan_in)), dtype=self.lora_A.dtype))

        resmat = self.layer_weights - self.scaling * lora_B @ torch.diag(vector_z) @lora_A 
        
        resmat = resmat.to(dtype)
        
        self.layer_weights.data = resmat
        del resmat, lora_A, lora_B, vector_z, weight
        
    def _dropout(self, A):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)
  
    # @torch.no_grad() # 这个地方的取消梯度很重要？ 好吧，非常他妈的重要！！！
    # 这里的逻辑有问题，干什么要每次都要重复加上这个结果呢，只需要最后加上就好了，我相当于反复的重复的加了SVD的值
    def sara_forward(self, X):
        torch_X_dtype = X.dtype #16
        
        diag_z = torch.diag(F.relu(self.vector_z)) # 32
        # print("self.scaling_factor init", self.scaling_factor)
        result = self.scaling_factor * self.scaling * self.lora_B @ diag_z @ self.lora_A
        
        result = result.to(torch_X_dtype) # 32 -> 16
        del diag_z
        
        return X + result
    
    def forward(self, X):
        return self.forward_fn(X)

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.shape        
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha, layer_module=layer
        )

    @classmethod
    def from_conv2d(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        # layer_weights = layer
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha, layer_module=layer
        )

    @classmethod
    def from_embedding(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
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




    
    # def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
    #     """
    #     Merge the active adapter weights into the base weights

    #     Args:
    #         safe_merge (`bool`, *optional*):
    #             If True, the merge operation will be performed in a copy of the original weights and check for NaNs
    #             before merging the weights. This is useful if you want to check if the merge operation will produce
    #             NaNs. Defaults to `False`.
    #         adapter_names (`list[str]`, *optional*):
    #             The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
    #             to `None`.
    #     """
    #     # adapter_names = check_adapters_to_merge(self, adapter_names)
    #     if not adapter_names:
    #         # no adapter to merge
    #         return

    #     for active_adapter in adapter_names:
    #         if active_adapter in self.lora_A.keys():
    #             base_layer = self.get_base_layer()
    #             if safe_merge:
    #                 # Note that safe_merge will be slower than the normal merge
    #                 # because of the copy operation.
    #                 orig_weights = base_layer.weight.data.clone()
    #                 delta_weight = self.get_delta_weight(active_adapter)
    #                 orig_weights = orig_weights + delta_weight
        
    #                 if not torch.isfinite(orig_weights).all():
    #                     raise ValueError(
    #                         f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
    #                     )
    #                 base_layer.weight.data = orig_weights
    #             else:
    #                 delta_weight = self.get_delta_weight(active_adapter)

    #                 base_layer.weight.data = base_layer.weight.data + delta_weight


    # def get_delta_weight(self) -> torch.Tensor:
    #     # 计算SVD的USV^T的值

    #     weight_A = self.lora_A.weight
    #     weight_B = self.lora_B.weight

    #     output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling

    #     self.lora_A.weight.data = weight_A
    #     self.lora_B.weight.data = weight_B

    #     return output_tensor