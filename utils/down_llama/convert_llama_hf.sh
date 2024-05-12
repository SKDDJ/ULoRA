# python ~/anaconda3/envs/ldm/lib/python3.8/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
#     --input_dir /root/shiym_proj/Sara/utils/down_llama/llama3/Meta-Llama-3-8B --model_size 8B --output_dir /root/shiym_proj/Sara/model/llama3/


python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /root/shiym_proj/Sara/utils/down_llama/llama3/Meta-Llama-3-8B  --model_size 8B --output_dir /root/shiym_proj/Sara/models/llama3_hf/ --llama_version 3
