"""
Phi-3 Text Generation Demo - macOS Optimized

macOS Adaptations:
1. device_map="mps" - Uses Metal Performance Shaders for Apple Silicon (M1/M2/M3)
   (Change to "cpu" for Intel Macs)
2. Local model loading - Models loaded from ./model/ directory instead of downloading each time
3. Simplified pipeline - Uses Hugging Face pipeline for easier inference on macOS

Requirements:
- Run `uv run python download_model.py` first to download the model locally
- Requires ~8GB RAM for model inference
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

# Define local model path
local_model_path = "./model/Phi-3-mini-4k-instruct"

# Load model and tokenizer from local directory
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="mps",  # Use Metal Performance Shaders for Apple Silicon (M1/M2/M3)
    torch_dtype="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Although we can now use the model and tokenizer directly, it's much easier to wrap it in a `pipeline` object:

from transformers import pipeline

# Create a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False
)

# Finally, we create our prompt as a user and give it to the model:

# The prompt (user input / query)
messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]

# Generate output
output = generator(messages)
print(output[0]["generated_text"])