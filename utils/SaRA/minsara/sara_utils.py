from minsara import SaRAParametrization
from torch import nn


def apply_to_sara(fn):
    """apply a function to SaRAParametrization layers, designed to be used with model.apply"""

    def apply_fn(layer):
        if isinstance(layer, SaRAParametrization):
            fn(layer)

    return apply_fn


enable_sara = lambda model: model.apply(apply_to_sara(lambda x: x.enable_sara()))
disable_sara = lambda model: model.apply(apply_to_sara(lambda x: x.disable_sara()))


# ------------------- helper function for collecting parameters for training/saving -------------------
def print_model_shapes(model):                    
    # 打印模型参数的名称和大小
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")
    # 打印模型参数的名称和数据类型
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")

def print_vector_parameters(model):
    classifier_params = 0
    trainable_params = 0
    vector_params = 0
    all_param = 0
    for n, param in model.named_parameters():
        num_params = param.numel() 
        all_param += num_params
        if 'original' in n:
            # print(f"{n} : {num_params:,d}\n")
            continue
        if param.requires_grad:
            if 'classifier' in n:
                classifier_params += num_params
            trainable_params += num_params
            if "vector_z" in n:
                vector_params += num_params
        # print(f"{n} : {num_params:,d}\n")
    print(
        f"vector params: {vector_params:,d} || trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    print(
        f"vector params: {vector_params:,d} || trainable params(wo classifier): {trainable_params-classifier_params:,d} || all params: {all_param-classifier_params:,d} || trainable%: {100 * (trainable_params-classifier_params) / (all_param-classifier_params)}"
    )
    return vector_params

def mark_only_sara_as_trainable(model):
    # 将除Sara之外的参数设为不需要梯度
    for param in model.parameters():
        param.requires_grad = False
    # 将Sara参数设为需要梯度
    for param in get_sara_params(model):
        param.requires_grad = True


def name_is_sara(name):
    return (
        len(name.split(".")) >= 4
        and (name.split(".")[-4]) == "parametrizations"
        and name.split(".")[-1] in ["vector_z","scaling_factor"]
        # and name.split(".")[-1] in ["lora_A", "lora_B","vector_z"]
    )

def name_is_bias(name):
    return name.split(".")[-1] == "bias"


def get_params_by_name(model, print_shapes=False, name_filter=None):
    for n, p in model.named_parameters():
        if name_filter is None or name_filter(n):
            if print_shapes:
                print(n, p.shape)
            yield p


def get_sara_params(model, print_shapes=False):
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_sara)
    


def get_bias_params(model, print_shapes=False):
    return get_params_by_name(model, print_shapes=print_shapes, name_filter=name_is_bias)


# def get_sara_state_dict(model):
#     return {k: v for k, v in model.state_dict().items() if name_is_sara(k)}

def get_sara_state_dict(model, original_state_dict):
    return {k: v for k, v in original_state_dict.items() if name_is_sara(k)}

def sara_state_dict(self):
    original_state_dict = type(self).state_dict(self)  # 获取原始 state_dict
    return get_sara_state_dict(self,original_state_dict)

# # ------------------- helper function for inferencing with multiple lora -------------------


# def _prepare_for_multiple_lora(lora_layer):
#     lora_layer.lora_As = []
#     lora_layer.lora_Bs = []


# def _append_lora(lora_layer):
#     lora_layer.lora_As.append(nn.Parameter(lora_layer.lora_A.clone()))
#     lora_layer.lora_Bs.append(nn.Parameter(lora_layer.lora_B.clone()))


# def load_multiple_lora(model, lora_state_dicts):
#     model.apply(apply_to_lora(_prepare_for_multiple_lora))
#     for state_dict in lora_state_dicts:
#         _ = model.load_state_dict(state_dict, strict=False)
#         model.apply(apply_to_lora(_append_lora))
#     return model


# def _select_lora(lora_layer, index):
#     lora_layer.lora_A = lora_layer.lora_As[index]
#     lora_layer.lora_B = lora_layer.lora_Bs[index]


# def select_lora(model, index):
#     model.apply(apply_to_lora(lambda x: _select_lora(x, index)))
#     return model


# # ------------------- helper function for tying and untieing weights -------------------


# def tie_weights(linear: nn.Linear, embedding: nn.Embedding):
#     """tie the weights of the linear layer and the embedding layer both with the same lora"""
#     # this line below is optional if the original is already tied
#     embedding.parametrizations.weight.original = linear.parametrizations.weight.original
#     embedding.parametrizations.weight[0].lora_A = linear.parametrizations.weight[0].lora_B
#     embedding.parametrizations.weight[0].lora_B = linear.parametrizations.weight[0].lora_A


# def untie_weights(linear: nn.Linear, embedding: nn.Embedding):
#     """untie the weights of the linear layer and the embedding layer"""
#     embedding.parametrizations.weight.original = nn.Parameter(embedding.weight.original.clone())
#     embedding.parametrizations.weight[0].lora_A = nn.Parameter(embedding.parametrizations.weight[0].lora_A.clone())
#     embedding.parametrizations.weight[0].lora_B = nn.Parameter(embedding.parametrizations.weight[0].lora_B.clone())
