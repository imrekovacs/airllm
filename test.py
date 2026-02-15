import os
from dotenv import load_dotenv

# load .env BEFORE importing airllm/huggingface so HF_HOME takes effect
load_dotenv(override=True)

from airllm import AutoModel

MODEL_CACHE_PATH = os.path.join(os.getenv("MODEL_CACHE_PATH", ".cache"), "Qwen2.5-Coder-7B")
HF_TOKEN = os.getenv("HF_TOKEN") or None

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AutoModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B",
    layer_shards_saving_path=MODEL_CACHE_PATH,
    hf_token=HF_TOKEN
)

input_text = [
        'What is the capital of United States?',
        #'I like',
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=False)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)
