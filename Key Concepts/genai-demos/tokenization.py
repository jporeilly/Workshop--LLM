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

def visualize_tokenization(text, filename):
    """Visualize how the text is broken down into tokens."""
    # Initialize the tokenizer (using the same encoding as GPT-3.5/4)
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Get tokens and token texts
    tokens = enc.encode(text)
    token_texts = [enc.decode([token]) for token in tokens]
    
    # Create visualization
    plt.figure(figsize=(15, 4))
    
    # Plot tokens as boxes
    for i, (token, text) in enumerate(zip(tokens, token_texts)):
        plt.plot([i, i+1, i+1, i, i], [0, 0, 1, 1, 0], 'b-')
        # Display token text and ID
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
        
        # Plot tokens for each text
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
    
    # Calculate token counts
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
    
    # Example texts for analysis
    texts = [
        "What is the capital of France?",
        "Tell me France's capital city",
        "Paris is located in which country?",
        "What is the capital of Germany?"
    ]
    
    # Demonstrate tokenization for a single text
    print("\nVisualizing tokenization for first text...")
    visualize_tokenization(texts[0], "single_text_tokenization.png")
    
    # Compare tokenization across different texts
    print("\nComparing tokenization across different texts...")
    compare_tokenization_variations(texts, "text_comparison.png")
    
    # Analyze token statistics
    print("\nAnalyzing token statistics...")
    analyze_token_stats(texts, "token_stats.png")
    
    # Additional examples to show interesting tokenization cases
    special_cases = [
        "OpenAI",
        "machine learning",
        "Python3.9",  # Removed emoji to avoid font issues
        "https://example.com",
        "Hello, world!"
    ]
    
    print("\nDemonstrating special tokenization cases...")
    compare_tokenization_variations(special_cases, "special_cases.png")
    
    print(f"\nAll plots have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    main()