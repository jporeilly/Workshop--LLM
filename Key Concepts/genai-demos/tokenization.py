import tiktoken
import pandas as pd
import matplotlib.pyplot as plt
import os

def explore_vocabulary(encoding_name="cl100k_base", n_samples=20):
    """Explore and visualize the tokenizer vocabulary."""
    enc = tiktoken.get_encoding(encoding_name)
    
    # Get the vocabulary dictionary
    # Note: This is a special case for tiktoken, as it doesn't expose the vocabulary directly
    vocab_dict = {}
    for i in range(100000):  # Sample a range of token IDs
        try:
            token_bytes = enc.decode_single_token_bytes(i)
            token_text = token_bytes.decode('utf-8', errors='replace')
            vocab_dict[i] = token_text
        except:
            continue
        if len(vocab_dict) >= n_samples:
            break
    
    print(f"\nSample of {encoding_name} vocabulary:")
    print("-" * 50)
    for token_id, token_text in list(vocab_dict.items())[:n_samples]:
        print(f"Token ID: {token_id:5d} | Token Text: '{token_text}'")

def analyze_token_mapping(text, encoding_name="cl100k_base"):
    """Analyze how text is mapped to tokens and back."""
    enc = tiktoken.get_encoding(encoding_name)
    
    # Get tokens
    tokens = enc.encode(text)
    
    print(f"\nToken mapping analysis for: '{text}'")
    print("-" * 50)
    print("Step 1: Text to Tokens")
    print(f"Original text: {text}")
    print(f"Token IDs: {tokens}")
    
    print("\nStep 2: Individual Token Analysis")
    for i, token in enumerate(tokens):
        token_text = enc.decode([token])
        print(f"Position {i+1}: Token ID {token:5d} → '{token_text}'")
    
    print("\nStep 3: Reconstruction")
    reconstructed = enc.decode(tokens)
    print(f"Reconstructed text: {reconstructed}")
    print(f"Matches original: {text == reconstructed}")

def compare_encodings():
    """Compare different tiktoken encodings."""
    sample_text = "OpenAI develops GPT-4, an advanced AI model!"
    encodings = [
        "cl100k_base",  # ChatGPT
        "p50k_base",    # GPT-3
        "r50k_base"     # Earlier models
    ]
    
    print("\nComparing different encodings:")
    print("-" * 50)
    for encoding_name in encodings:
        enc = tiktoken.get_encoding(encoding_name)
        tokens = enc.encode(sample_text)
        print(f"\n{encoding_name}:")
        print(f"Number of tokens: {len(tokens)}")
        print("Token breakdown:")
        for token in tokens:
            print(f"  {token:5d} → '{enc.decode([token])}'")

def main():
    # Explore vocabulary
    print("\nExploring tokenizer vocabulary...")
    explore_vocabulary()
    
    # Analyze specific examples
    print("\nAnalyzing token mappings...")
    examples = [
        "OpenAI",
        "machine learning",
        "https://example.com",
        "Python3.9",
        "Hello, world!"
    ]
    
    for example in examples:
        analyze_token_mapping(example)
    
    # Compare different encodings
    print("\nComparing different encodings...")
    compare_encodings()

if __name__ == "__main__":
    main()