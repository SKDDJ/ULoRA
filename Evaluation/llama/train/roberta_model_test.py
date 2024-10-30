from transformers import AutoTokenizer, AutoModelForMaskedLM
from labml.logger import inspect
import torch.nn as nn

# Load model
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
model = AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-base")

# Inspect layers that contain 'norm' in their name
for name, param in model.named_parameters():
    if 'output.dense' in name:
        # Split the name to traverse the model's submodules
        components = name.split('.')
        submodule = model
        for comp in components[:-1]:  # Traverse until the second last component
            submodule = getattr(submodule, comp)
        
        # Get the layer/module
        layer = getattr(submodule, components[-1])
        
        # Get the exact type of the layer
        layer_type = type(layer)
        
        # Inspect the layer name and its type
        inspect(name, layer_type)


# from transformers import AutoTokenizer, AutoModelForMaskedLM
# import torch.nn as nn
# from labml.logger import inspect

# # Load model
# tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
# model = AutoModelForMaskedLM.from_pretrained("FacebookAI/roberta-base")

# # Inspect layers that contain 'norm' in their name
# for name, param in model.named_parameters():
#     if 'norm' in name:
#         # Split the name to traverse the model's submodules
#         components = name.split('.')
#         submodule = model
#         for comp in components[:-1]:  # Traverse until the second last component
#             submodule = getattr(submodule, comp)
        
#         # Get the layer/module
#         layer = getattr(submodule, components[-1])
        
#         # Check if the layer is an instance of nn.Linear
#         is_linear = isinstance(layer, nn.Linear)
        
#         # Inspect the layer name and the check result
#         inspect(name, is_linear)
