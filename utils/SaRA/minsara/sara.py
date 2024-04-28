import math
from functools import partial

import torch
from torch import svd_lowrank
import torch.nn.utils.parametrize as parametrize
from torch import nn
import lightning as pl


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

class SaRAParametrization(pl.LightningModule):
    def __init__(self, fan_in, fan_out, fan_in_fan_out=False, rank=4, lora_dropout_p=0.0, lora_alpha=8, layer_weights=None):
        super().__init__() 
        # if weight is stored as (fan_out, fan_in), the memory layout of A & B follows (W + BA)x
        # otherwise, it's x(W + AB). This allows us to tie the weights between linear layers and embeddings
        self.layer_weights = layer_weights # pass in the layer(nn.Linear(in, out)) to use svd to 
        # initialize A and B
        # 保存layer权重的副本，后续操作基于此副本
        # todo test the params count
        # self.count = 0
        # print(f"SaRAParametrization count: {self.count}")
        # self.count += 1
        
        # self.layer_weights_copy = layer_weights.clone()

        self.swap = (lambda x: (x[1], x[0])) if fan_in_fan_out else (lambda x: x)
        
        self.lora_alpha, self.rank = lora_alpha, rank
        self.scaling = lora_alpha / rank
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.lora_dropout_p = lora_dropout_p
        # 初始化参数但不在此处更新权重
        # self.initialize_params()
        # 注册一个前向传播前的钩子，以自动更新权重一次
        self._forward_pre_hook_handle = self.register_forward_pre_hook(self._update_weights_once)
        # 添加一个额外的属性来标记是否更新过权重
        self._updated = False

        # todo: test removing the code below
        self.forward_fn = self.sara_forward

    def _update_weights_once(self, *args):
        if not self._updated:  # 只有在未更新权重的情况下才执行
            self.update_weights()
            self._forward_pre_hook_handle.remove()  # 移除钩子
            self._updated = True  # 更新标记，防止再次执
        # 权重更新后立即移除钩子，确保只执行一次
        
    def update_weights(self):
        # my impelementation of sara
        
        # weight = self.layer_weights_copy
        weight = self.layer_weights
        # print("weight ", weight)
        # print(weight.data)
        # print("self.layer_weights shape: ", self.layer_weights.shape)
        # self.layer_weights shape:  torch.Size([7, 5])
        # exit()
        r = self.rank
        
        # todo test init_lora_weights
        init_lora_weights = "pissa_niter_64"
        # todo check if it is need to del teh U V S
        if init_lora_weights == "pissa":
                V, S, Uh = torch.linalg.svd(weight, full_matrices=False)
                Vr = V[:, : r]
                Sr = S[: r]
                Sr /= self.scaling
                Uhr = Uh[: r]
        elif len(init_lora_weights.split("_niter_")) == 2:                    
                    Vr, Sr, Ur = svd_lowrank(
                        weight, r, niter=int(init_lora_weights.split("_niter_")[-1])
                    )
                    Sr /= self.scaling
                    Uhr = Ur.t()
                    
                    
        lora_A = Uhr
        lora_B = Vr
        # vector_z = torch.diag(Sr)
        vector_z = Sr

        # initialization of A and B
        # note: because we may use weight tying, so we have to define the lora_X as nn.Parameter not the nn.Linear
        self.lora_A = nn.Parameter(torch.zeros(self.swap((self.rank, self.fan_in))))
        self.lora_B = nn.Parameter(torch.zeros(self.swap((self.fan_out, self.rank))))
        self.vector_z = nn.Parameter(vector_z)
        # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        

        self.lora_A.data = lora_A
        self.lora_B.data = lora_B
        self.vector_z.data = vector_z     
        
        
        # drop out which won't be used 
        self.lora_dropout = nn.Dropout(p=self.lora_dropout_p) if self.lora_dropout_p > 0 else lambda x: x
        self.dropout_fn = self._dropout if self.lora_dropout_p > 0 else lambda x: x
        self.register_buffer("lora_dropout_mask", torch.ones(self.swap((1, self.fan_in)), dtype=self.lora_A.dtype))
        # """
        # lora_A shape:  torch.Size([2, 5])
        # lora_B shape:  torch.Size([7, 2])
        # vector_z shape:  torch.Size([2])
        # """
        # print("lora_A shape: ", self.lora_A.shape)
        # print("lora_B shape: ", self.lora_B.shape)
        # print("vector_z shape: ", self.vector_z.shape)
        # print("=====================================")
        # print("lora_A: ", lora_A)
        # print("lora_B: ", lora_B)
        # print("vector_z: ", vector_z)
        # print("=====================================")
        # print("weight shape: ", weight.shape)
        # weight shape:  torch.Size([7, 5])
        
        # because here we don't nanipulate the input but the model weights so the in,out is opposite to the input shape

        # 这里初始化参数，但不应用于layer的权重
        # 将原先在 __init__ 中进行的权重更新逻辑移至此方法
        # print("check the svd getted vector_Z")
        # print(vector_z)
        
        resmat = weight - self.scaling * lora_B @ torch.diag(vector_z) @lora_A 
        # print("the original svd product")
        # print("self.scaling * lora_B @ torch.diag(vector_z) @lora_A", self.scaling * lora_B @ torch.diag(vector_z) @lora_A)
        # print("resmat shape: ", resmat.shape) # resmat shape:  torch.Size([7, 5])
        # print("resmat: ", resmat)
        # todo !!! need to update the res matrix here
        # todo need to update the res matrix
        # self.layer.weight.data = resmat
        self.layer_weights = resmat
        # print("scaling factor: ", self.scaling) # 0.5
        # print("lora_alpha: ", self.lora_alpha) # 1
        # print("rank: ", self.rank) # 2
        
        # print("original matrix")
        # print(weight)
        # print("\n")
        # print("residual matrix")
        # print(resmat)
        
        # print("weight shape: ", self.layer_weights.shape)
        # weight = self.layer.weight

        # 假设这里是更新权重的代码
        # Example:
        # weight.data = updated_weight_logic

        # 最终，将更新的权重赋值回去
        # self.layer.weight.data = weight.data

    def _dropout(self, A):
        # to mimic the original implementation: A @ dropout(x), we do (A * dropout(ones)) @ x
        return A * self.lora_dropout(self.lora_dropout_mask)

    def sara_forward(self, X):
        # here the X is not the input but the layer or the original weights (in, out)
        diag_z = torch.diag(self.vector_z)
        # print("X is what: ", X)
        # print("check forward vector Z")
        # print(self.vector_z)
        # 将lora_A, diag_z和lora_B进行矩阵乘法操作
        # here we use @ rather than torch.matmul to make the code more readable
        result = self.lora_B @ diag_z @ self.lora_A
        # result = torch.matmul(*self.swap(torch.matmul(self.lora_B, diag_z), self.dropout_fn(self.lora_A)))
        # 根据需要改变结果的形状，以匹配X的形状，并进行必要的缩放
        # 注：如果result的形状以及与X兼容，则无需调整形状
        # 他妈的这里的X是原始矩阵，但是实际上我需要后来的残差矩阵作为输入
        # processed = result.view(X.shape) * self.scaling
        processed = result.view(self.layer_weights.shape) * self.scaling
        # print("here we can get the product of the lora_A, diag_z, lora_B")
        # print("and we can know the svd product(USV^T) is : ", processed)
        
        # print("after merge: ", self.layer_weights+processed)
        # 将处理后的结果与输入resmat相加，返回最终结果
        return self.layer_weights + processed
        # return X + torch.matmul(*self.swap((self.lora_B, self.dropout_fn(self.lora_A)))).view(X.shape) * self.scaling

    def forward(self, X):
        return self.forward_fn(X)

    def disable_sara(self):
        self.forward_fn = lambda x: x

    def enable_sara(self):
        self.forward_fn = self.sara_forward

    @classmethod
    def from_linear(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=8):
        """使用 `cls` 来创建和返回类的实例"""
        fan_out, fan_in = layer.weight.shape
        # 传入layer权重的本身
        layer_weights = layer.weight.data # note 就是这里，在这里传入最终正确了！！！
        # layer_weights_copy = layer.weight.data.clone()
        
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha, layer_weights=layer_weights
        )

    @classmethod
    def from_conv2d(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_out, fan_in = layer.weight.view(layer.weight.shape[0], -1).shape
        layer_weights = layer.weight.data
        return cls(
            fan_in, fan_out, fan_in_fan_out=False, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha, layer_weights=layer_weights
        )

    @classmethod
    def from_embedding(cls, layer, rank=4, lora_dropout_p=0.0, lora_alpha=1):
        fan_in, fan_out = layer.weight.shape
        layer_weights = layer.weight.data
        return cls(
            fan_in, fan_out, fan_in_fan_out=True, rank=rank, lora_dropout_p=lora_dropout_p, lora_alpha=lora_alpha, layer_weights=layer_weights
        )

"""  - **子键** (`"weight"`): 表示要对 `nn.Linear` 层的权重进行参数化。
  - **子值** (`partial(SaRAParametrization.from_linear, rank=8)`):
    - `SaRAParametrization.from_linear` 是一个设计用来对线性层的权重执行特定变换的参数化函数。"""
default_sara_config = {  # specify which layers to add sara to, by default only add to linear layers
    nn.Linear: {
        "weight": partial(SaRAParametrization.from_linear, rank=2),
    },
}



def apply_sara(layer, register=True, merge=False, sara_config=default_sara_config):
    #    这行定义了一个函数`apply_sara`，它接受一个网络层（`layer`），三个可选参数`register`（默认为True），
    # `merge`（默认为False），和`sara_config`（默认为`default_sara_config`）。

    """add sara parametrization to a layer, designed to be used with model.apply"""
    #    这是一个文档字符串，解释了这个函数的用途：给一个层添加SaRA参数化，通常用于与`model.apply`一起使用。

    if register:#    这个条件判断是检查是否需要注册SaRA参数化。
        
        if type(layer) in sara_config:#    如果当前层的类型在`sara_config`中定义了相应的SaRA参数化设置，则继续执行。
            # print(sara_config[type(layer)])#    {'weight': functools.partial(<class '__main__.SaRAParametrization'>, rank=8)}
            for attr_name, parametrization in sara_config[type(layer)].items():
                # if not hasattr(layer.parametrizations, attr_name):  # 检查是否已经存在参数化
                    # print(f"{attr_name} 没有参数化")
                    # print("没有参数化")
                #    遍历该层类型在`sara_config`中定义的所有SaRA参数化。
                # attr_name:"weight"; parametrization: partial(SaRAParametrization.from_linear, rank=8) 
                # 一个`partial`对象，包含了一个参数化函数和一些参数。
                    """`register_parametrization` 是用来在 PyTorch 框架中对特定模块的张量（通常是一个网络层的权重或偏置）注册一个参数化操作。实际上，参数化指的是对张量应用一个可训练的变换，通常用于实现复杂的正则化、约束或改良模型架构的目的。"""
                    parametrize.register_parametrization(layer, attr_name, parametrization(layer))
                #    对每个属性和参数化，调用`register_parametrization`来在层上注册SaRA参数化。

                
    else:  # this will remove all parametrizations, use with caution
    #    如果`register`为False，则进入这个分支，这个分支将移除所有参数化。

        if hasattr(layer, "parametrizations"):#    检查层是否有`parametrizations`属性。

            for attr_name in layer.parametrizations.keys():#    如果有，遍历所有的参数化属性。
                """函数 `remove_parametrizations` 在神经网络模型中用于移除被参数化（或变换）的特定张量（通常是网络层的权重或偏执）。这个函数允许用户根据需要决定在移除参数化后，张量保持原始状态还是保持应用参数化后的状态。"""
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
    """Add SaRA parameterization to specific layers in a model by names"""
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




