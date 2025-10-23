from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    revision="main",
    local_dir="./Llama-3.2-11B-Vision-Instruct",
    ignore_patterns=["original/*"]  # 如果你不想下载原始检查点
)

print("Downloaded to:", local_dir)