"""
Chapter 2 - Tokens and Token Embeddings Demo - macOS Optimized

This script demonstrates 5 key concepts from Chapter 2:
1. LLM Tokenization and Text Generation
2. Comparing Different Tokenizers
3. Contextualized Word Embeddings (BERT-style)
4. Sentence Embeddings (for semantic search)
5. Static Word Embeddings and Applications (Word2Vec, GloVe)

macOS Adaptations:
- device_map="mps" for Apple Silicon (M1/M2/M3)
- All models loaded from local ./model/ directory
- Optimized for macOS inference

Requirements:
- Run `uv run python download_model.py` first to download all models
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
from gensim.models import Word2Vec
import pandas as pd
from urllib import request
import numpy as np


def part1_llm_tokenization():
    """
    Part 1: Downloading and Running An LLM

    Demonstrates:
    - Loading a language model and tokenizer
    - Text generation
    - Understanding tokenization (text -> token IDs -> tokens)
    """
    print("\n" + "=" * 70)
    print("PART 1: LLM Tokenization and Text Generation")
    print("=" * 70)

    # Load model and tokenizer from local directory
    print("\n[1/3] Loading Phi-3 model and tokenizer from local directory...")
    model_path = "./model/Phi-3-mini-4k-instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="mps",  # Use MPS for Apple Silicon
        torch_dtype="auto",
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✓ Model loaded successfully")

    # Generate text
    print("\n[2/3] Generating text...")
    prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("mps")
    generation_output = model.generate(
        input_ids=input_ids,
        max_new_tokens=50
    )

    generated_text = tokenizer.decode(generation_output[0])
    print(f"\nGenerated text:\n{generated_text}\n")

    # Demonstrate tokenization
    print("[3/3] Understanding tokenization...")
    print(f"\nOriginal prompt: {prompt[:50]}...")
    print(f"\nToken IDs: {input_ids[0][:10].tolist()}...")

    print("\nIndividual tokens:")
    for i, token_id in enumerate(input_ids[0][:10]):
        token_text = tokenizer.decode(token_id)
        print(f"  Token {i}: ID={token_id.item():5d} → '{token_text}'")

    print("\n✓ Part 1 Complete")


def part2_compare_tokenizers():
    """
    Part 2: Comparing Trained LLM Tokenizers

    Demonstrates:
    - How different tokenizers split text differently
    - Case sensitivity differences
    - Handling of special characters, emojis, and code
    """
    print("\n" + "=" * 70)
    print("PART 2: Comparing Different Tokenizers")
    print("=" * 70)

    # Test text with various challenges
    test_text = "English CAPITALIZATION show_tokens False None elif == >= 12.0*50=600"

    # Tokenizers to compare (using local models)
    tokenizer_configs = [
        ("bert-base-uncased", "./model/bert-base-uncased"),
        ("bert-base-cased", "./model/bert-base-cased"),
        ("gpt2", "./model/gpt2"),
        ("Phi-3", "./model/Phi-3-mini-4k-instruct"),
    ]

    print(f"\nTest text: {test_text}\n")

    for name, path in tokenizer_configs:
        print(f"\n--- {name} ---")
        tokenizer = AutoTokenizer.from_pretrained(path)
        token_ids = tokenizer(test_text).input_ids
        tokens = [tokenizer.decode(tid) for tid in token_ids]

        print(f"Tokens ({len(tokens)}): {' | '.join(tokens)}")

    print("\n✓ Part 2 Complete")


def part3_contextualized_embeddings():
    """
    Part 3: Contextualized Word Embeddings From a Language Model (Like BERT)

    Demonstrates:
    - Getting contextualized embeddings for each token
    - Each token gets a vector that depends on context
    - Shape: [batch_size, num_tokens, embedding_dim]
    """
    print("\n" + "=" * 70)
    print("PART 3: Contextualized Word Embeddings")
    print("=" * 70)

    print("\n[1/2] Loading DeBERTa model...")
    # Load tokenizer and model from local directories
    tokenizer = AutoTokenizer.from_pretrained("./model/deberta-base")
    model = AutoModel.from_pretrained("./model/deberta-v3-xsmall")
    model = model.to("mps")  # Move to MPS device
    print("✓ Model loaded")

    print("\n[2/2] Getting contextualized embeddings...")
    # Tokenize and get embeddings
    text = "Hello world"
    tokens = tokenizer(text, return_tensors='pt')
    tokens = {k: v.to("mps") for k, v in tokens.items()}

    output = model(**tokens)[0]

    print(f"\nInput text: '{text}'")
    print(f"Output shape: {output.shape}")
    print(f"  - Batch size: {output.shape[0]}")
    print(f"  - Number of tokens: {output.shape[1]}")
    print(f"  - Embedding dimension: {output.shape[2]}")

    # Show tokens
    print("\nTokens:")
    for i, token_id in enumerate(tokens['input_ids'][0]):
        token_text = tokenizer.decode(token_id)
        embedding_vector = output[0, i, :5].tolist()  # First 5 dimensions
        print(f"  Token {i}: '{token_text}' → [{embedding_vector[0]:.4f}, {embedding_vector[1]:.4f}, ...]")

    print("\n✓ Part 3 Complete")


def part4_sentence_embeddings():
    """
    Part 4: Text Embeddings (For Sentences and Whole Documents)

    Demonstrates:
    - Converting entire sentences to single vectors
    - Used for semantic search and similarity
    - Shape: [embedding_dim]
    """
    print("\n" + "=" * 70)
    print("PART 4: Sentence Embeddings")
    print("=" * 70)

    print("\n[1/2] Loading sentence transformer model...")
    model = SentenceTransformer('./model/all-mpnet-base-v2')
    print("✓ Model loaded")

    print("\n[2/2] Encoding sentences...")
    sentences = [
        "Best movie ever!",
        "This film was amazing!",
        "I love programming in Python.",
    ]

    embeddings = model.encode(sentences)

    print(f"\nEncoded {len(sentences)} sentences")
    print(f"Embedding shape per sentence: {embeddings[0].shape}")

    # Show sentences and their embeddings
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        print(f"\nSentence {i+1}: '{sentence}'")
        print(f"  Embedding (first 5 dims): [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")

    # Calculate similarity between first two sentences
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"\nSimilarity between sentence 1 and 2: {similarity:.4f}")

    similarity_1_3 = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
    print(f"Similarity between sentence 1 and 3: {similarity_1_3:.4f}")
    print("(Higher similarity = more semantically similar)")

    print("\n✓ Part 4 Complete")


def part5_word_embeddings():
    """
    Part 5: Word Embeddings Beyond LLMs

    Demonstrates:
    - Static word embeddings (GloVe)
    - Word similarity using pre-trained embeddings
    - Application: Song recommendation using Word2Vec
    """
    print("\n" + "=" * 70)
    print("PART 5: Static Word Embeddings and Applications")
    print("=" * 70)

    # Part 5a: GloVe word embeddings
    print("\n--- Part 5a: GloVe Word Embeddings ---")
    print("Loading GloVe embeddings (cached by gensim)...")
    glove_model = api.load("glove-wiki-gigaword-50")
    print("✓ GloVe embeddings loaded")

    # Find similar words
    test_word = "king"
    similar_words = glove_model.most_similar([glove_model[test_word]], topn=10)

    print(f"\nWords most similar to '{test_word}':")
    for word, similarity in similar_words:
        print(f"  {word:15s} (similarity: {similarity:.4f})")

    # Part 5b: Song recommendation system
    print("\n\n--- Part 5b: Song Recommendation System ---")
    print("Downloading playlist dataset...")

    try:
        # Get playlist data
        data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')
        lines = data.read().decode("utf-8").split('\n')[2:]
        playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

        # Get song metadata
        songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
        songs_file = songs_file.read().decode("utf-8").split('\n')
        songs = [s.rstrip().split('\t') for s in songs_file]
        songs_df = pd.DataFrame(data=songs, columns=['id', 'title', 'artist'])
        songs_df = songs_df.set_index('id')

        print(f"✓ Loaded {len(playlists)} playlists")

        # Train Word2Vec model
        print("Training Word2Vec model on playlists...")
        w2v_model = Word2Vec(
            playlists,
            vector_size=32,
            window=20,
            negative=50,
            min_count=1,
            workers=4
        )
        print("✓ Model trained")

        # Make recommendations
        print("\n--- Song Recommendations ---")

        # Example 1: Metallica
        song_id = 2172
        song_info = songs_df.iloc[song_id]
        print(f"\nInput song: '{song_info['title']}' by {song_info['artist']}")

        similar_song_ids = np.array(
            w2v_model.wv.most_similar(positive=str(song_id), topn=5)
        )[:, 0]

        print("Recommended songs:")
        recommendations = songs_df.iloc[similar_song_ids]
        for idx, row in recommendations.iterrows():
            print(f"  • '{row['title']}' by {row['artist']}")

        # Example 2: 2Pac
        song_id_2 = 842
        song_info_2 = songs_df.iloc[song_id_2]
        print(f"\nInput song: '{song_info_2['title']}' by {song_info_2['artist']}")

        similar_song_ids_2 = np.array(
            w2v_model.wv.most_similar(positive=str(song_id_2), topn=5)
        )[:, 0]

        print("Recommended songs:")
        recommendations_2 = songs_df.iloc[similar_song_ids_2]
        for idx, row in recommendations_2.iterrows():
            print(f"  • '{row['title']}' by {row['artist']}")

    except Exception as e:
        print(f"⚠ Could not download playlist data: {e}")
        print("Skipping song recommendation demo")

    print("\n✓ Part 5 Complete")


def main():
    """Run all 5 parts of Chapter 2 demos"""
    print("\n" + "=" * 70)
    print("Chapter 2 - Tokens and Token Embeddings")
    print("Hands-On Large Language Models - macOS Edition")
    print("=" * 70)

    print("\nThis demo covers 5 key concepts:")
    print("1. LLM Tokenization and Text Generation")
    print("2. Comparing Different Tokenizers")
    print("3. Contextualized Word Embeddings")
    print("4. Sentence Embeddings")
    print("5. Static Word Embeddings and Applications")

    # Run all parts
    try:
        part1_llm_tokenization()
    except Exception as e:
        print(f"\n⚠ Error in Part 1: {e}")

    try:
        part2_compare_tokenizers()
    except Exception as e:
        print(f"\n⚠ Error in Part 2: {e}")

    try:
        part3_contextualized_embeddings()
    except Exception as e:
        print(f"\n⚠ Error in Part 3: {e}")

    try:
        part4_sentence_embeddings()
    except Exception as e:
        print(f"\n⚠ Error in Part 4: {e}")

    try:
        part5_word_embeddings()
    except Exception as e:
        print(f"\n⚠ Error in Part 5: {e}")

    print("\n" + "=" * 70)
    print("All demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
