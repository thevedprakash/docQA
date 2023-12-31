import sys
import os


from huggingface_hub import hf_hub_download

model_path = 'models'

# Check if the folder exists
if not os.path.exists(model_path):
    # If it does not exist, create it
    os.makedirs(model_path)
    print(f"The directory {model_path} was created.")
else:
    print(f"The directory {model_path} already exists.")
# repo_id = sys.argv[0]
# filename = sys.argv[1]

repo_id="TheBloke/Llama-2-7B-Chat-GGUF"
filename="llama-2-7b-chat.Q8_0.gguf"

hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path ,local_dir_use_symlinks=False)
