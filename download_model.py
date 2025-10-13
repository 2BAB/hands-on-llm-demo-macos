"""
Download all models required for the Hands-On LLM book chapters.

Models are downloaded in parallel using multiple threads for faster completion.
This script downloads models used in:
- Chapter 1: Basic text generation with Phi-3
- Chapter 2: Tokens, embeddings, and various tokenizers
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


def check_model_exists(local_dir):
    """Check if a model already exists locally."""
    if not os.path.exists(local_dir):
        return False
    # Check for config.json which indicates a valid model
    config_path = os.path.join(local_dir, "config.json")
    return os.path.exists(config_path)


def check_tokenizer_exists(local_dir):
    """Check if a tokenizer already exists locally."""
    if not os.path.exists(local_dir):
        return False
    # Check for tokenizer config or vocab files
    tokenizer_config = os.path.join(local_dir, "tokenizer_config.json")
    vocab_json = os.path.join(local_dir, "vocab.json")
    vocab_txt = os.path.join(local_dir, "vocab.txt")
    tokenizer_json = os.path.join(local_dir, "tokenizer.json")

    return (os.path.exists(tokenizer_config) or
            os.path.exists(vocab_json) or
            os.path.exists(vocab_txt) or
            os.path.exists(tokenizer_json))


def download_causal_lm(model_name, local_dir):
    """Download a causal language model and its tokenizer."""
    try:
        if check_model_exists(local_dir):
            print(f"[{model_name}] ⊙ Already exists, skipping download")
            return f"⊙ {model_name} (skipped)"

        print(f"[{model_name}] ⟳ Downloading model and tokenizer...")
        os.makedirs(local_dir, exist_ok=True)

        # Download model
        print(f"[{model_name}]   → Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=False,
        )
        model.save_pretrained(local_dir)

        # Download tokenizer
        print(f"[{model_name}]   → Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)

        print(f"[{model_name}] ✓ Download complete")
        return f"✓ {model_name}"
    except Exception as e:
        error_msg = f"✗ {model_name}: {str(e)}"
        print(f"[{model_name}] ✗ Error: {str(e)}")
        traceback.print_exc()
        return error_msg


def download_tokenizer_only(model_name, local_dir):
    """Download only the tokenizer (for comparison purposes)."""
    try:
        if check_tokenizer_exists(local_dir):
            print(f"[{model_name}] ⊙ Already exists, skipping download")
            return f"⊙ {model_name} (tokenizer, skipped)"

        print(f"[{model_name}] ⟳ Downloading tokenizer...")
        os.makedirs(local_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)

        print(f"[{model_name}] ✓ Download complete")
        return f"✓ {model_name} (tokenizer)"
    except Exception as e:
        error_msg = f"✗ {model_name}: {str(e)}"
        print(f"[{model_name}] ✗ Error: {str(e)}")
        return error_msg


def download_auto_model(model_name, local_dir):
    """Download a general AutoModel (like DeBERTa)."""
    try:
        if check_model_exists(local_dir):
            print(f"[{model_name}] ⊙ Already exists, skipping download")
            return f"⊙ {model_name} (skipped)"

        print(f"[{model_name}] ⟳ Downloading model and tokenizer...")
        os.makedirs(local_dir, exist_ok=True)

        print(f"[{model_name}]   → Downloading model weights...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(local_dir)

        print(f"[{model_name}]   → Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)

        print(f"[{model_name}] ✓ Download complete")
        return f"✓ {model_name}"
    except Exception as e:
        error_msg = f"✗ {model_name}: {str(e)}"
        print(f"[{model_name}] ✗ Error: {str(e)}")
        return error_msg


def download_sentence_transformer(model_name, local_dir):
    """Download a sentence transformer model."""
    try:
        # Check if model exists by looking for config files
        if os.path.exists(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
            print(f"[{model_name}] ⊙ Already exists, skipping download")
            return f"⊙ {model_name} (skipped)"

        print(f"[{model_name}] ⟳ Downloading sentence transformer model...")
        os.makedirs(local_dir, exist_ok=True)

        model = SentenceTransformer(model_name)
        model.save(local_dir)

        print(f"[{model_name}] ✓ Download complete")
        return f"✓ {model_name}"
    except Exception as e:
        error_msg = f"✗ {model_name}: {str(e)}"
        print(f"[{model_name}] ✗ Error: {str(e)}")
        return error_msg


def download_gensim_embeddings(model_name):
    """Download word embeddings from gensim."""
    try:
        # Gensim handles caching automatically
        print(f"[{model_name}] ⟳ Checking/downloading gensim embeddings...")
        print(f"[{model_name}]   (gensim will use cached version if available)")
        model = api.load(model_name)
        print(f"[{model_name}] ✓ Ready to use")
        return f"✓ {model_name} (gensim)"
    except Exception as e:
        error_msg = f"✗ {model_name}: {str(e)}"
        print(f"[{model_name}] ✗ Error: {str(e)}")
        return error_msg


def main():
    print("=" * 70)
    print("Model Download Script - Hands-On LLM for macOS")
    print("=" * 70)
    print("\nThis will download all models required for the book chapters.")
    print("Downloads will run in parallel for faster completion.\n")

    # Define all models to download
    tasks = [
        # Chapter 1 & 2: Main model
        ("causal_lm", "microsoft/Phi-3-mini-4k-instruct", "./model/Phi-3-mini-4k-instruct"),

        # Chapter 2: Tokenizers for comparison
        ("tokenizer", "bert-base-uncased", "./model/bert-base-uncased"),
        ("tokenizer", "bert-base-cased", "./model/bert-base-cased"),
        ("tokenizer", "gpt2", "./model/gpt2"),
        ("tokenizer", "google/flan-t5-small", "./model/flan-t5-small"),
        ("tokenizer", "Xenova/gpt-4", "./model/gpt-4-tokenizer"),
        ("tokenizer", "bigcode/starcoder2-15b", "./model/starcoder2-15b"),
        ("tokenizer", "facebook/galactica-1.3b", "./model/galactica-1.3b"),

        # Chapter 2: DeBERTa for contextualized embeddings
        ("auto_model", "microsoft/deberta-v3-xsmall", "./model/deberta-v3-xsmall"),
        ("tokenizer", "microsoft/deberta-base", "./model/deberta-base"),

        # Chapter 2: Sentence transformer for text embeddings
        ("sentence_transformer", "sentence-transformers/all-mpnet-base-v2", "./model/all-mpnet-base-v2"),

        # Chapter 2: Word embeddings (gensim - cached automatically)
        ("gensim", "glove-wiki-gigaword-50", None),
    ]

    print(f"Total models to download: {len(tasks)}\n")

    results = []

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        for task_type, model_name, local_dir in tasks:
            if task_type == "causal_lm":
                future = executor.submit(download_causal_lm, model_name, local_dir)
            elif task_type == "tokenizer":
                future = executor.submit(download_tokenizer_only, model_name, local_dir)
            elif task_type == "auto_model":
                future = executor.submit(download_auto_model, model_name, local_dir)
            elif task_type == "sentence_transformer":
                future = executor.submit(download_sentence_transformer, model_name, local_dir)
            elif task_type == "gensim":
                future = executor.submit(download_gensim_embeddings, model_name)

            futures.append(future)

        # Wait for all downloads to complete
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(f"✗ Error: {str(e)}")

    # Print summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    for result in results:
        print(result)

    print("\n✓ All downloads complete!")
    print("You can now run the chapter scripts using local models.")


if __name__ == "__main__":
    main()