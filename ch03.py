"""
Chapter 3 - Looking Inside Transformer LLMs - macOS Optimized

This script demonstrates 3 key concepts from Chapter 3:
1. Loading and inspecting LLM architecture
2. Understanding model inputs/outputs and token generation
3. KV cache optimization for faster generation

macOS Adaptations:
- device_map="mps" for Apple Silicon (M1/M2/M3)
- All models loaded from local ./model/ directory
- Optimized for macOS inference

Requirements:
- Run `uv run python download_model.py` first to download Phi-3 model
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import time


def part1_load_and_inspect_llm():
    """
    Part 1: Loading the LLM and Inspecting Its Architecture

    Demonstrates:
    - Loading a transformer-based LLM
    - Inspecting the model architecture
    - Understanding model components (embeddings, layers, attention, MLP, etc.)
    """
    print("\n" + "=" * 70)
    print("PART 1: Loading and Inspecting LLM Architecture")
    print("=" * 70)

    print("\n[1/3] Loading Phi-3 model from local directory...")
    model_path = "./model/Phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="mps",  # Use MPS for Apple Silicon
        torch_dtype="auto",
        trust_remote_code=False,
    )
    print("✓ Model loaded successfully")

    print("\n[2/3] Creating text generation pipeline...")
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=50,
        do_sample=False,
    )
    print("✓ Pipeline created")

    print("\n[3/3] Generating sample text...")
    prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
    output = generator(prompt)
    
    print(f"\nPrompt: {prompt}")
    print(f"\nGenerated text:\n{output[0]['generated_text']}")

    print("\n" + "-" * 70)
    print("Model Architecture Overview:")
    print("-" * 70)
    print(model)

    print("\n✓ Part 1 Complete")
    
    return model, tokenizer


def part2_inputs_outputs_token_generation(model, tokenizer):
    """
    Part 2: The Inputs and Outputs of a Trained Transformer LLM

    Demonstrates:
    - How text input becomes token IDs
    - Model's internal processing (embeddings → transformer layers → lm_head)
    - How probability distribution is converted to next token (argmax sampling)
    """
    print("\n" + "=" * 70)
    print("PART 2: Model Inputs, Outputs, and Token Generation")
    print("=" * 70)

    prompt = "The capital of France is"
    print(f"\nPrompt: '{prompt}'")

    print("\n[1/4] Tokenizing input...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to("mps")
    
    print(f"Input token IDs: {input_ids[0].tolist()}")
    print(f"Input shape: {input_ids.shape} (batch_size=1, sequence_length={input_ids.shape[1]})")

    print("\n[2/4] Running through transformer layers (model.model)...")
    model_output = model.model(input_ids)
    
    print(f"Model output shape: {model_output[0].shape}")
    print(f"  - Batch size: {model_output[0].shape[0]}")
    print(f"  - Sequence length: {model_output[0].shape[1]}")
    print(f"  - Hidden dimension: {model_output[0].shape[2]}")

    print("\n[3/4] Running through language model head (lm_head)...")
    lm_head_output = model.lm_head(model_output[0])
    
    print(f"LM head output shape: {lm_head_output.shape}")
    print(f"  - Batch size: {lm_head_output.shape[0]}")
    print(f"  - Sequence length: {lm_head_output.shape[1]}")
    print(f"  - Vocabulary size: {lm_head_output.shape[2]}")

    print("\n[4/4] Selecting next token (greedy decoding)...")
    # Get the last token's logits
    next_token_logits = lm_head_output[0, -1]
    
    # Get the token with highest probability (argmax)
    next_token_id = next_token_logits.argmax(-1)
    next_token_text = tokenizer.decode(next_token_id)
    
    print(f"Next token ID: {next_token_id.item()}")
    print(f"Next token text: '{next_token_text}'")
    print(f"\nComplete output: '{prompt} {next_token_text}'")

    # Show top 5 predictions
    print("\nTop 5 predicted tokens:")
    top_5_ids = torch.topk(next_token_logits, k=5).indices
    for i, token_id in enumerate(top_5_ids):
        token_text = tokenizer.decode(token_id)
        prob = torch.softmax(next_token_logits, dim=-1)[token_id].item()
        print(f"  {i+1}. '{token_text}' (probability: {prob:.4f})")

    print("\n✓ Part 2 Complete")


def part3_kv_cache_optimization(model, tokenizer):
    """
    Part 3: Speeding up Generation by Caching Keys and Values

    Demonstrates:
    - KV cache for faster generation
    - Performance comparison: with vs without KV cache
    - Why caching saves computation
    """
    print("\n" + "=" * 70)
    print("PART 3: KV Cache Optimization")
    print("=" * 70)

    prompt = "Write a very long email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."
    
    print(f"\nPrompt: {prompt[:60]}...")
    print("\nGenerating 100 tokens with different cache settings...")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to("mps")

    # Test with cache enabled
    print("\n[1/2] Testing WITH KV cache (use_cache=True)...")
    torch.mps.synchronize()  # Ensure all operations are complete
    start_time = time.time()
    
    generation_output_cached = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        use_cache=True
    )
    
    torch.mps.synchronize()
    cached_time = time.time() - start_time
    
    cached_text = tokenizer.decode(generation_output_cached[0])
    print(f"✓ Generation complete in {cached_time:.2f} seconds")
    print(f"Generated {len(generation_output_cached[0]) - len(input_ids[0])} tokens")

    # Test without cache
    print("\n[2/2] Testing WITHOUT KV cache (use_cache=False)...")
    torch.mps.synchronize()
    start_time = time.time()
    
    generation_output_uncached = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        use_cache=False
    )
    
    torch.mps.synchronize()
    uncached_time = time.time() - start_time
    
    print(f"✓ Generation complete in {uncached_time:.2f} seconds")
    print(f"Generated {len(generation_output_uncached[0]) - len(input_ids[0])} tokens")

    # Performance comparison
    print("\n" + "-" * 70)
    print("Performance Comparison:")
    print("-" * 70)
    print(f"WITH cache:    {cached_time:.2f} seconds")
    print(f"WITHOUT cache: {uncached_time:.2f} seconds")
    speedup = uncached_time / cached_time
    print(f"Speedup:       {speedup:.2f}x faster with cache")
    print(f"Time saved:    {uncached_time - cached_time:.2f} seconds ({(1 - 1/speedup)*100:.1f}% reduction)")

    print("\n" + "-" * 70)
    print("Why KV Cache Helps:")
    print("-" * 70)
    print("• WITHOUT cache: Recomputes attention for ALL previous tokens at each step")
    print("• WITH cache:    Reuses cached key/value matrices from previous tokens")
    print("• Result:        O(n²) → O(n) complexity reduction")
    print("-" * 70)

    print("\nGenerated text (first 200 chars):")
    print(cached_text[:200] + "...")

    print("\n✓ Part 3 Complete")


def main():
    """Run all 3 parts of Chapter 3 demos"""
    print("\n" + "=" * 70)
    print("Chapter 3 - Looking Inside Transformer LLMs")
    print("Hands-On Large Language Models - macOS Edition")
    print("=" * 70)

    print("\nThis demo covers 3 key concepts:")
    print("1. Loading and Inspecting LLM Architecture")
    print("2. Model Inputs, Outputs, and Token Generation")
    print("3. KV Cache Optimization")

    # Part 1: Load and inspect
    try:
        model, tokenizer = part1_load_and_inspect_llm()
    except Exception as e:
        print(f"\n⚠ Error in Part 1: {e}")
        import traceback
        traceback.print_exc()
        return

    # Part 2: Inputs/outputs
    try:
        part2_inputs_outputs_token_generation(model, tokenizer)
    except Exception as e:
        print(f"\n⚠ Error in Part 2: {e}")
        import traceback
        traceback.print_exc()

    # Part 3: KV cache
    try:
        part3_kv_cache_optimization(model, tokenizer)
    except Exception as e:
        print(f"\n⚠ Error in Part 3: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
