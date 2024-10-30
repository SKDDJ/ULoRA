from huggingface_hub import HfApi
api = HfApi()

# huggingface-cli upload-large-folder HuggingFaceM4/Docmatix --repo-type=dataset /path/to/local/docmatix --num-workers=16

api.upload_large_folder(
    folder_path="/root/shiym_proj/Sara/vit-finetune/data",
    repo_id="Shiym/ViT-FineTune",
    repo_type="dataset",
    num_workers=16
)