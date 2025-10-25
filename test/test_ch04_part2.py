"""
Test only PART 2 (FLAN-T5) of Chapter 4 to verify the fix.
This runs much faster than the full ch04.py script.
"""

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report
from datasets import load_dataset
from tqdm import tqdm


def evaluate_performance(y_true, y_pred):
    """Create and print the classification report."""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)


def main():
    print("=" * 70)
    print("PART 2: Testing FLAN-T5 Classification (Quick Verification)")
    print("=" * 70)

    print("\n[1/4] Loading dataset...")
    data = load_dataset("rotten_tomatoes")
    print(f"✓ Loaded {len(data['test'])} test samples")

    print("\n[2/4] Loading FLAN-T5 model...")
    pipe = pipeline(
        "text2text-generation",
        model="./model/flan-t5-small",
        device="mps"
    )
    print("✓ Model loaded")

    print("\n[3/4] Preparing prompts and running inference...")
    prompt = "Is the following sentence positive or negative? "
    data_with_prompts = data.map(lambda example: {"t5": prompt + example['text']})

    y_pred = []
    for output in tqdm(pipe(KeyDataset(data_with_prompts["test"], "t5")),
                       total=len(data_with_prompts["test"])):
        text = output[0]["generated_text"].lower()
        y_pred.append(0 if text == "negative" else 1)

    print(f"✓ Generated {len(y_pred)} predictions")

    print("\n[4/4] Evaluating performance...")
    evaluate_performance(data["test"]["label"], y_pred)

    print("\n" + "=" * 70)
    print("Expected results (from original notebook):")
    print("  Accuracy: ~84%")
    print("  Precision/Recall: 0.83-0.85 for both classes")
    print("=" * 70)


if __name__ == "__main__":
    main()
