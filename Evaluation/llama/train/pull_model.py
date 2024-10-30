# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Shiym/llama2-7B", revision="a349e52dd90708e4c086f9db2f9007caef9a33ea")
model = AutoModelForCausalLM.from_pretrained("Shiym/llama2-7B", revision="a349e52dd90708e4c086f9db2f9007caef9a33ea")