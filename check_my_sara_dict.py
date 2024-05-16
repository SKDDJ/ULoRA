from safetensors import safe_open

# Specify the path to your .safetensors file
file_path = "/root/shiym_proj/Sara/llama-lora/r-128-4-alpha-512-qv-bs-128-lr-3e-2-len-256-epochs-3-seed-42/model/model.safetensors"

# Read the .safetensors file
tensors = {}
with safe_open(file_path, framework="pt", device=0) as f:  # Adjust 'framework' and 'device' as necessary
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

# Print the keys and shapes of the tensors
print("Keys and shapes of the tensors in the safetensors file:")
for key, tensor in tensors.items():
    print(f"{key}: shape {tensor.shape}")