"""
Chapter 4 - Text Classification - macOS Optimized

This script demonstrates 3 key classification approaches from Chapter 4:
1. Text Classification with Representation Models
   - Task-specific models (RoBERTa sentiment classifier)
   - Supervised classification with embeddings
   - Zero-shot classification
2. Classification with Generative Models
   - Encoder-decoder models (FLAN-T5)
   - ChatGPT API (optional)

macOS Adaptations:
- device_map="mps" for Apple Silicon (M1/M2/M3)
- All models loaded from local ./model/ directory
- Optimized for macOS inference

Requirements:
- Run `uv run python download_model.py` first to download all models
- For ChatGPT section: Set OPENAI_API_KEY environment variable (optional)
"""

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


def load_data():
    """Load the Rotten Tomatoes dataset."""
    print("\n" + "=" * 70)
    print("Loading Rotten Tomatoes Dataset")
    print("=" * 70)

    print("\nDownloading dataset from Hugging Face...")
    data = load_dataset("rotten_tomatoes")

    print(f"\n✓ Dataset loaded successfully")
    print(f"  - Train samples: {len(data['train'])}")
    print(f"  - Validation samples: {len(data['validation'])}")
    print(f"  - Test samples: {len(data['test'])}")

    print("\nSample data:")
    print(f"  Text: {data['train'][0]['text'][:80]}...")
    print(f"  Label: {data['train'][0]['label']} (0=negative, 1=positive)")

    return data


def evaluate_performance(y_true, y_pred):
    """Create and print the classification report."""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)


def part1_task_specific_model(data):
    """
    Part 1a: Using a Task-specific Model

    Demonstrates:
    - Using a pre-trained sentiment classification model (RoBERTa)
    - Running inference on test data
    - Evaluating performance
    """
    print("\n" + "=" * 70)
    print("PART 1a: Task-specific Sentiment Classification (RoBERTa)")
    print("=" * 70)

    print("\n[1/3] Loading RoBERTa sentiment model from local directory...")
    model_path = "./model/twitter-roberta-base-sentiment-latest"

    pipe = pipeline(
        "text-classification",  # Explicitly specify task
        model=model_path,
        tokenizer=model_path,
        top_k=None,  # Return all scores
        device="mps"  # Use MPS for Apple Silicon
    )
    print("✓ Model loaded successfully")

    print("\n[2/3] Running inference on test set...")
    print("(This may take a few minutes)")

    y_pred = []
    for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
        # Output format: [{'label': 'negative', 'score': ...}, {'label': 'neutral', 'score': ...}, {'label': 'positive', 'score': ...}]
        # We need to extract negative and positive scores
        scores_dict = {item['label']: item['score'] for item in output}
        negative_score = scores_dict.get('negative', 0)
        positive_score = scores_dict.get('positive', 0)
        assignment = np.argmax([negative_score, positive_score])
        y_pred.append(assignment)

    print(f"\n✓ Inference complete")
    print(f"Generated {len(y_pred)} predictions")

    print("\n[3/3] Evaluating performance...")
    evaluate_performance(data["test"]["label"], y_pred)

    print("\n✓ Part 1a Complete")


def part2_supervised_classification(data):
    """
    Part 1b: Supervised Classification with Embeddings

    Demonstrates:
    - Converting text to embeddings using sentence transformers
    - Training a logistic regression classifier
    - Making predictions on test data
    """
    print("\n" + "=" * 70)
    print("PART 1b: Supervised Classification with Embeddings")
    print("=" * 70)

    print("\n[1/4] Loading sentence transformer model...")
    model = SentenceTransformer('./model/all-mpnet-base-v2')
    print("✓ Model loaded")

    print("\n[2/4] Converting text to embeddings...")
    print("  → Encoding training set...")
    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    print("  → Encoding test set...")
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

    print(f"\n✓ Embeddings created")
    print(f"  Training embeddings shape: {train_embeddings.shape}")
    print(f"  Test embeddings shape: {test_embeddings.shape}")

    print("\n[3/4] Training Logistic Regression classifier...")
    clf = LogisticRegression(random_state=42)
    clf.fit(train_embeddings, data["train"]["label"])
    print("✓ Classifier trained")

    print("\n[4/4] Making predictions and evaluating...")
    y_pred = clf.predict(test_embeddings)
    evaluate_performance(data["test"]["label"], y_pred)

    print("\n✓ Part 1b Complete")

    return test_embeddings


def part3_zero_shot_classification(data, test_embeddings):
    """
    Part 1c: Zero-shot Classification

    Demonstrates:
    - Creating embeddings for label descriptions
    - Using cosine similarity for classification
    - No training required
    """
    print("\n" + "=" * 70)
    print("PART 1c: Zero-shot Classification")
    print("=" * 70)

    print("\n[1/3] Loading sentence transformer model...")
    model = SentenceTransformer('./model/all-mpnet-base-v2')
    print("✓ Model loaded")

    print("\n[2/3] Creating embeddings for label descriptions...")
    label_descriptions = ["A negative review", "A positive review"]
    label_embeddings = model.encode(label_descriptions)

    print(f"  Label 0 (negative): '{label_descriptions[0]}'")
    print(f"  Label 1 (positive): '{label_descriptions[1]}'")
    print(f"✓ Label embeddings created")

    print("\n[3/3] Classifying using cosine similarity...")
    # Find the best matching label for each document
    sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
    y_pred = np.argmax(sim_matrix, axis=1)

    print("✓ Classification complete")
    print("\nEvaluation:")
    evaluate_performance(data["test"]["label"], y_pred)

    print("\n" + "-" * 70)
    print("💡 Tip: Try different label descriptions!")
    print("   - 'A very negative movie review' / 'A very positive movie review'")
    print("   - 'Bad movie' / 'Good movie'")
    print("   - Experiment to see how it affects performance")
    print("-" * 70)

    print("\n✓ Part 1c Complete")


def part4_average_embeddings_classification(data):
    """
    Part 1d: Classification by Averaging Target Embeddings

    Demonstrates:
    - Alternative approach without explicit classifier
    - Average embeddings per class
    - Use cosine similarity to predict
    """
    print("\n" + "=" * 70)
    print("PART 1d: Classification by Averaging Target Embeddings")
    print("=" * 70)

    print("\n[1/3] Loading sentence transformer and creating embeddings...")
    model = SentenceTransformer('./model/all-mpnet-base-v2')

    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)
    print("✓ Embeddings created")

    print("\n[2/3] Averaging embeddings per class...")
    # Combine embeddings with labels
    df = pd.DataFrame(
        np.hstack([train_embeddings, np.array(data["train"]["label"]).reshape(-1, 1)])
    )

    # Average the embeddings of all documents in each target label
    averaged_target_embeddings = df.groupby(768).mean().values  # Column 768 is the label

    print(f"✓ Averaged embeddings shape: {averaged_target_embeddings.shape}")
    # Convert to numpy array for comparison
    labels = np.array(data['train']['label'])
    print(f"  Class 0 (negative): averaged from {(labels == 0).sum()} samples")
    print(f"  Class 1 (positive): averaged from {(labels == 1).sum()} samples")

    print("\n[3/3] Finding best matching embeddings...")
    # Find the best matching embeddings between test documents and target embeddings
    sim_matrix = cosine_similarity(test_embeddings, averaged_target_embeddings)
    y_pred = np.argmax(sim_matrix, axis=1)

    print("✓ Classification complete")
    print("\nEvaluation:")
    evaluate_performance(data["test"]["label"], y_pred)

    print("\n✓ Part 1d Complete")


def part5_encoder_decoder_generation(data):
    """
    Part 2: Classification with Encoder-Decoder Generative Models

    Demonstrates:
    - Using FLAN-T5 for text classification via generation
    - Prompting strategy for classification
    - Converting generated text to predictions
    """
    print("\n" + "=" * 70)
    print("PART 2: Classification with Encoder-Decoder Models (FLAN-T5)")
    print("=" * 70)

    print("\n[1/4] Loading FLAN-T5 model from local directory...")
    model_path = "./model/flan-t5-small"

    pipe = pipeline(
        "text2text-generation",
        model=model_path,
        device="mps"
    )
    print("✓ Model loaded successfully")

    print("\n[2/4] Preparing prompts...")
    prompt = "Is the following sentence positive or negative? "

    # Add prompts to dataset
    data_with_prompts = data.map(lambda example: {"t5": prompt + example['text']})

    print(f"Example prompt:")
    print(f"  '{data_with_prompts['test'][0]['t5'][:100]}...'")

    print("\n[3/4] Running inference on test set...")
    print("(This may take a few minutes)")

    y_pred = []
    for output in tqdm(pipe(KeyDataset(data_with_prompts["test"], "t5")), total=len(data_with_prompts["test"])):
        text = output[0]["generated_text"].lower()
        # Map generated text to labels
        y_pred.append(0 if text == "negative" else 1)

    print(f"\n✓ Inference complete")
    print(f"Generated {len(y_pred)} predictions")

    print("\n[4/4] Evaluating performance...")
    evaluate_performance(data["test"]["label"], y_pred)

    print("\n✓ Part 2 Complete")


def part6_chatgpt_classification(data):
    """
    Part 3: Classification with ChatGPT (Optional)

    Demonstrates:
    - Using OpenAI API for classification
    - Prompt engineering for classification tasks
    - API-based inference

    Note: Requires OPENAI_API_KEY environment variable
    """
    print("\n" + "=" * 70)
    print("PART 3: Classification with ChatGPT (Optional)")
    print("=" * 70)

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠ OPENAI_API_KEY not found in environment variables")
        print("  To use this feature:")
        print("  1. Get an API key from https://platform.openai.com/")
        print("  2. Set it: export OPENAI_API_KEY='your-key-here'")
        print("  3. Re-run this script")
        print("\n✓ Part 3 Skipped")
        return

    try:
        import openai
    except ImportError:
        print("\n⚠ openai package not installed")
        print("  Run: uv add openai")
        print("\n✓ Part 3 Skipped")
        return

    print("\n[1/3] Setting up OpenAI client...")
    client = openai.OpenAI(api_key=api_key)
    print("✓ Client created")

    print("\n[2/3] Defining prompt template...")
    prompt_template = """Predict whether the following document is a positive or negative movie review:

[DOCUMENT]

If it is positive return 1 and if it is negative return 0. Do not give any other answers.
"""

    # Test with a single example first
    print("\nTesting with a single example...")
    test_doc = data["test"]["text"][0]
    print(f"  Document: '{test_doc[:80]}...'")

    def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
        """Generate an output based on a prompt and an input document."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.replace("[DOCUMENT]", document)}
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0
        )
        return chat_completion.choices[0].message.content

    try:
        result = chatgpt_generation(prompt_template, test_doc)
        print(f"  Prediction: {result}")
        print("✓ API test successful")
    except Exception as e:
        print(f"\n⚠ API test failed: {e}")
        print("✓ Part 3 Skipped")
        return

    print("\n[3/3] Running on full test set...")
    print("⚠ WARNING: This will use API credits!")
    print(f"  - Total test samples: {len(data['test'])}")
    print(f"  - Estimated cost: Check OpenAI pricing")

    user_input = input("\nProceed with full test set? (y/n): ")
    if user_input.lower() != 'y':
        print("✓ Part 3 Skipped by user")
        return

    print("\nRunning predictions (this may take 10-15 minutes)...")
    predictions = []
    for doc in tqdm(data["test"]["text"]):
        try:
            pred = chatgpt_generation(prompt_template, doc)
            predictions.append(pred)
        except Exception as e:
            print(f"\n⚠ Error: {e}")
            predictions.append("0")  # Default to negative on error

    # Extract predictions
    y_pred = [int(pred) if pred in ['0', '1'] else 0 for pred in predictions]

    print("\n✓ Predictions complete")
    print("\nEvaluation:")
    evaluate_performance(data["test"]["label"], y_pred)

    print("\n✓ Part 3 Complete")


def main():
    """Run all parts of Chapter 4 demos."""
    print("\n" + "=" * 70)
    print("Chapter 4 - Text Classification")
    print("Hands-On Large Language Models - macOS Edition")
    print("=" * 70)

    print("\nThis demo covers text classification with:")
    print("1. Representation Models")
    print("   a. Task-specific models (RoBERTa)")
    print("   b. Supervised classification with embeddings")
    print("   c. Zero-shot classification")
    print("   d. Average embeddings classification")
    print("2. Generative Models")
    print("   a. Encoder-decoder models (FLAN-T5)")
    print("   b. ChatGPT API (optional)")

    # Load data once
    try:
        data = load_data()
    except Exception as e:
        print(f"\n⚠ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Part 1a: Task-specific model
    try:
        part1_task_specific_model(data)
    except Exception as e:
        print(f"\n⚠ Error in Part 1a: {e}")
        import traceback
        traceback.print_exc()

    # Part 1b: Supervised classification
    try:
        test_embeddings = part2_supervised_classification(data)
    except Exception as e:
        print(f"\n⚠ Error in Part 1b: {e}")
        import traceback
        traceback.print_exc()
        test_embeddings = None

    # Part 1c: Zero-shot classification
    if test_embeddings is not None:
        try:
            part3_zero_shot_classification(data, test_embeddings)
        except Exception as e:
            print(f"\n⚠ Error in Part 1c: {e}")
            import traceback
            traceback.print_exc()

    # Part 1d: Average embeddings
    try:
        part4_average_embeddings_classification(data)
    except Exception as e:
        print(f"\n⚠ Error in Part 1d: {e}")
        import traceback
        traceback.print_exc()

    # Part 2: FLAN-T5
    try:
        part5_encoder_decoder_generation(data)
    except Exception as e:
        print(f"\n⚠ Error in Part 2: {e}")
        import traceback
        traceback.print_exc()

    # Part 3: ChatGPT (optional)
    try:
        part6_chatgpt_classification(data)
    except Exception as e:
        print(f"\n⚠ Error in Part 3: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
