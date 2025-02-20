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
    
    # Save vocabulary sample to a text file
    vocab_file = os.path.join(output_dir, "vocabulary_sample.txt")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write(f"Sample of {encoding_name} vocabulary:\n")
        f.write("-" * 50 + "\n")
        for token_id, token_text in list(vocab_dict.items())[:n_samples]:
            f.write(f"Token ID: {token_id:5d} | Token Text: '{token_text}'\n")
    
    print(f"Vocabulary sample saved to: {vocab_file}")

def analyze_token_mapping(text, output_dir, encoding_name="cl100k_base"):
    """Analyze how text is mapped to tokens and back."""
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    
    # Save analysis to a text file
    analysis_file = os.path.join(output_dir, f"token_mapping_{text[:20]}.txt")
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
        reconstructed = enc.decode(tokens)
        f.write(f"Reconstructed text: {reconstructed}\n")
        f.write(f"Matches original: {text == reconstructed}\n")
    
    print(f"Token mapping analysis saved to: {analysis_file}")

def visualize_tokenization(text, output_dir, filename):
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
    save_plot(plt, output_dir, filename)

def compare_tokenization_variations(texts, output_dir, filename):
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
    save_plot(plt, output_dir, filename)

def analyze_token_stats(texts, output_dir, filename):
    """Analyze and visualize tokenization statistics."""
    enc = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(enc.encode(text)) for text in texts]
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(texts)), token_counts)
    plt.xticks(range(len(texts)), [textwrap.fill(t, 20) for t in texts], rotation=45)
    plt.ylabel('Number of Tokens')
    plt.title('Token Count Comparison')
    plt.tight_layout()
    save_plot(plt, output_dir, filename)

def compare_encodings(output_dir):
    """Compare different tiktoken encodings."""
    sample_text = "OpenAI develops GPT-4, an advanced AI model!"
    encodings = [
        "cl100k_base",  # ChatGPT
        "p50k_base",    # GPT-3
        "r50k_base"     # Earlier models
    ]
    
    # Save comparison to a text file
    comparison_file = os.path.join(output_dir, "encoding_comparison.txt")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("Comparing different encodings:\n")
        f.write("-" * 50 + "\n")
        for encoding_name in encodings:
            enc = tiktoken.get_encoding(encoding_name)
            tokens = enc.encode(sample_text)
            f.write(f"\n{encoding_name}:\n")
            f.write(f"Number of tokens: {len(tokens)}\n")
            f.write("Token breakdown:\n")
            for token in tokens:
                f.write(f"  {token:5d} → '{enc.decode([token])}'\n")
    
    print(f"Encoding comparison saved to: {comparison_file}")

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
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Basic text examples
    texts = [
        "What is the capital of France?",
        "Tell me France's capital city",
        "Paris is located in which country?",
        "What is the capital of Germany?"
    ]
    
    visualize_tokenization(texts[0], output_dir, "single_text_tokenization.png")
    compare_tokenization_variations(texts, output_dir, "text_comparison.png")
    analyze_token_stats(texts, output_dir, "token_stats.png")
    
    # Special cases visualization
    compare_tokenization_variations(examples, output_dir, "special_cases.png")
    
    # Compare different encodings
    compare_encodings(output_dir)
    
    print(f"\nAll analysis results have been saved to: {output_dir}")

if __name__ == "__main__":
    main()