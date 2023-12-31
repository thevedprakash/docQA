import sys
from huggingface_hub import hf_hub_download


# repo_id = sys.argv[0]
# filename = sys.argv[1]

repo_id="TheBloke/Llama-2-7B-Chat-GGUF"
filename="llama-2-7b-chat.Q8_0.gguf"

hf_hub_download(repo_id=repo_id, filename=filename)
'/Users/ved/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/llama-2-7b-chat.Q8_0.gguf'