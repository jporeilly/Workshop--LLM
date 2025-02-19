import numpy as np
import matplotlib.pyplot as plt
import tiktoken
import textwrap
from sklearn.decomposition import PCA
import os

# Create output directory
output_dir = "tokenization_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def explore_vocabulary(encoding_name="cl100k_base", n_samples=20):
    """Explore and visualize the tokenizer vocabulary."""
    enc = tiktoken.get_encoding(encoding_name)
    
    # Get the vocabulary dictionary
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

def visualize_tokenization(text, filename):
    """Visualize how the text is broken down into tokens."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    token_texts = [enc.decode([token]) for token in tokens]
    
    plt.figure(figsize=(15, 4))
    for i, (token, text) in enumerate(zip(tokens, token_texts)):
        plt.plot([i, i+1, i+1, i, i], [0, 0, 1, 1, 0], 'b-')
        plt.text(i + 0.5, 0.5, f'"{text}"', ha='center', va='center')
        plt.text(i + 0.5, -0.2, str(token), ha='center', va='center', color='red')
    
    plt.xlim(-0.2, len(tokens) + 0.2)
    plt.ylim(-0.5, 1.5)
    plt.title('Text Tokenization Visualization')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def compare_tokenization_variations(texts, filename):
    """Compare tokenization of similar texts."""
    enc = tiktoken.get_encoding("cl100k_base")
    plt.figure(figsize=(15, len(texts) * 2))
    
    for idx, text in enumerate(texts):
        tokens = enc.encode(text)
        token_texts = [enc.decode([token]) for token in tokens]
        
        for i, (token, token_text) in enumerate(zip(tokens, token_texts)):
            plt.plot([i, i+1, i+1, i, i], 
                    [idx, idx, idx+1, idx+1, idx], 'b-')
            plt.text(i + 0.5, idx + 0.5, f'"{token_text}"', 
                    ha='center', va='center', fontsize=8)
            plt.text(i + 0.5, idx + 0.2, str(token), 
                    ha='center', va='center', color='red', fontsize=6)
    
    plt.yticks(np.arange(len(texts)) + 0.5, texts)
    plt.title('Comparison of Tokenization Across Similar Texts')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def analyze_token_stats(texts, filename):
    """Analyze and visualize tokenization statistics."""
    enc = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(enc.encode(text)) for text in texts]
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(texts)), token_counts)
    plt.xticks(range(len(texts)), [textwrap.fill(t, 20) for t in texts], rotation=45)
    plt.ylabel('Number of Tokens')
    plt.title('Token Count Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

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
    print(f"\nPlots will be saved to the '{output_dir}' directory.")
    
    # Explore vocabulary first
    print("\nExploring tokenizer vocabulary...")
    explore_vocabulary()
    
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
        analyze_token_mapping(example)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Basic text examples
    texts = [
        "What is the capital of France?",
        "Tell me France's capital city",
        "Paris is located in which country?",
        "What is the capital of Germany?"
    ]
    
    visualize_tokenization(texts[0], "single_text_tokenization.png")
    compare_tokenization_variations(texts, "text_comparison.png")
    analyze_token_stats(texts, "token_stats.png")
    
    # Special cases visualization
    compare_tokenization_variations(examples, "special_cases.png")
    
    # Compare different encodings
    compare_encodings()
    
    print(f"\nAll plots have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main()