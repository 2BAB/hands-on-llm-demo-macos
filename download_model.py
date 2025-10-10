from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Define the local directory
local_model_dir = "./model/Phi-3-mini-4k-instruct"

# Create directory if it doesn't exist
os.makedirs(local_model_dir, exist_ok=True)

print(f"Downloading model to {local_model_dir}...")

# Download and save the model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=False,
)
model.save_pretrained(local_model_dir)

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer.save_pretrained(local_model_dir)

print(f"Model and tokenizer saved to {local_model_dir}")
print("You can now use this local path in your scripts.")