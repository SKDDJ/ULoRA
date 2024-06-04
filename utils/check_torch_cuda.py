import torch

print(torch.__version__)
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

mat1 = torch.randn(2,3).to(0)
mat2 = torch.randn(3,3).to(0)

y = torch.mm(mat1, mat2)
print(y)