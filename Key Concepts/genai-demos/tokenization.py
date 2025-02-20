import numpy as np
import matplotlib.pyplot as plt
import tiktoken
import textwrap
from sklearn.decomposition import PCA
import os
from datetime import datetime

def ensure_output_directory():
    """Create and return the output directory path with timestamp."""
    base_dir = "tokenization_analysis"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"analysis_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def save_plot(plt, output_dir, filename):
    """Save the current plot to the visualizations directory."""
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path)
    print(f"Saved visualization to: {full_path}")
    plt.close()

def explore_vocabulary(output_dir, encoding_name="cl100k_base", n_samples=20):
    """Explore and visualize the tokenizer vocabulary."""
    enc = tiktoken.get_encoding(encoding_name)
    
    print(f"\nSample of {encoding_name} vocabulary:")
    print("-" * 50)
    
    # Get the vocabulary dictionary
    vocab_dict = {}
    for i in range(100000):  # Sample a range of token IDs
        try:
            token_bytes = enc.decode_single_token_bytes(i)
            token_text = token_bytes.decode('utf-8', errors='replace')
            vocab_dict[i] = token_text
            if len(vocab_dict) < n_samples:
                print(f"Token ID: {i:5d} | Token Text: '{token_text}'")
        except:
            continue
        if len(vocab_dict) >= n_samples:
            break
    
    # Save vocabulary sample to a text file
    vocab_file = os.path.join(output_dir, "vocabulary_sample.txt")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write(f"Sample of {encoding_name} vocabulary:\n")
        f.write("-" * 50 + "\n")
        for token_id, token_text in list(vocab_dict.items())[:n_samples]:
            f.write(f"Token ID: {token_id:5d} | Token Text: '{token_text}'\n")

def analyze_token_mapping(text, output_dir, encoding_name="cl100k_base"):
    """Analyze how text is mapped to tokens and back."""
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    
    print(f"\nToken mapping analysis for: '{text}'")
    print("-" * 50)
    
    # Print to console
    print("Step 1: Text to Tokens")
    print(f"Original text: {text}")
    print(f"Token IDs: {tokens}\n")
    
    print("Step 2: Individual Token Analysis")
    for i, token in enumerate(tokens):
        token_text = enc.decode([token])
        print(f"Position {i+1}: Token ID {token:5d} → '{token_text}'")
    
    print("\nStep 3: Reconstruction")
    reconstructed = enc.decode(tokens)
    print(f"Reconstructed text: {reconstructed}")
    print(f"Matches original: {text == reconstructed}")
    
    # Save to file
    safe_filename = "".join(c if c.isalnum() else "_" for c in text[:20])
    analysis_file = os.path.join(output_dir, f"token_mapping_{safe_filename}.txt")
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write(f"Token mapping analysis for: '{text}'\n")
        f.write("-" * 50 + "\n")
        f.write("Step 1: Text to Tokens\n")
        f.write(f"Original text: {text}\n")
        f.write(f"Token IDs: {tokens}\n\n")
        
        f.write("Step 2: Individual Token Analysis\n")
        for i, token in enumerate(tokens):
            token_text = enc.decode([token])
            f.write(f"Position {i+1}: Token ID {token:5d} → '{token_text}'\n")
        
        f.write("\nStep 3: Reconstruction\n")
        f.write(f"Reconstructed text: {reconstructed}\n")
        f.write(f"Matches original: {text == reconstructed}\n")

def compare_encodings(output_dir):
    """Compare different tiktoken encodings."""
    sample_text = "OpenAI develops GPT-4, an advanced AI model!"
    encodings = [
        "cl100k_base",  # ChatGPT
        "p50k_base",    # GPT-3
        "r50k_base"     # Earlier models
    ]
    
    print("\nComparing different encodings:")
    print("-" * 50)
    
    # Print to console and save to file
    comparison_file = os.path.join(output_dir, "encoding_comparison.txt")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("Comparing different encodings:\n")
        f.write("-" * 50 + "\n")
        
        for encoding_name in encodings:
            enc = tiktoken.get_encoding(encoding_name)
            tokens = enc.encode(sample_text)
            
            # Print to console
            print(f"\n{encoding_name}:")
            print(f"Number of tokens: {len(tokens)}")
            print("Token breakdown:")
            for token in tokens:
                print(f"  {token:5d} → '{enc.decode([token])}'")
            
            # Write to file
            f.write(f"\n{encoding_name}:\n")
            f.write(f"Number of tokens: {len(tokens)}\n")
            f.write("Token breakdown:\n")
            for token in tokens:
                f.write(f"  {token:5d} → '{enc.decode([token])}'\n")

def main():
    # Create output directory with timestamp
    output_dir = ensure_output_directory()
    print(f"\nAnalysis results will be saved to: {output_dir}")
    
    # Explore vocabulary first
    print("\nExploring tokenizer vocabulary...")
    explore_vocabulary(output_dir)
    
    # Example texts for analysis
    examples = [
        "OpenAI",
        "machine learning",
        "https://example.com",
        "Python3.9",
        "Hello, world!"
    ]
    
    # Analyze each example
    for example in examples:
        analyze_token_mapping(example, output_dir)
    
    # Compare different encodings
    compare_encodings(output_dir)
    
    print(f"\nAll analysis results have been saved to: {output_dir}")

if __name__ == "__main__":
    main()