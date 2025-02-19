import numpy as np
import matplotlib.pyplot as plt
import tiktoken
import textwrap
from sklearn.decomposition import PCA
import os

# Create output directory if it doesn't exist
output_dir = "tokenization_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def visualize_tokenization(text, filename):
    """Visualize how the text is broken down into tokens and save to file."""
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

def print_token_analysis(text):
    """Print detailed token analysis for a text."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    token_texts = [enc.decode([token]) for token in tokens]
    
    print(f"\nText: {text}")
    print("Tokens:")
    for i, (token, token_text) in enumerate(zip(tokens, token_texts)):
        print(f"  {i+1}. Token ID: {token:5d} | Token Text: '{token_text}'")
    print(f"Total tokens: {len(tokens)}")
    print("-" * 50)

def compare_tokenization_variations(texts, filename):
    """Compare tokenization of similar texts and save to file."""
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
    """Analyze and visualize tokenization statistics and save to file."""
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

def main():
    print(f"\nPlots will be saved to the '{output_dir}' directory.")
    
    # Basic text examples
    texts = [
        "What is the capital of France?",
        "Tell me France's capital city",
        "Paris is located in which country?",
        "What is the capital of Germany?"
    ]
    
    print("\nVisualizing tokenization for first text...")
    visualize_tokenization(texts[0], "single_text_tokenization.png")
    
    print("\nComparing tokenization across different texts...")
    compare_tokenization_variations(texts, "text_comparison.png")
    
    print("\nAnalyzing token statistics...")
    analyze_token_stats(texts, "token_stats.png")
    
    # Special cases with detailed token analysis
    special_cases = [
        "OpenAI",
        "machine learning",
        "special_token",
        "https://example.com",
        "Python3.9"
    ]
    
    print("\nDemonstrating special tokenization cases...")
    compare_tokenization_variations(special_cases, "special_cases.png")
    
    print("\nDetailed token analysis for special cases:")
    for text in special_cases:
        print_token_analysis(text)
    
    print(f"\nAll plots have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main()