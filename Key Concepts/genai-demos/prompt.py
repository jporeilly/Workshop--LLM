from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import textwrap
import os
from getpass import getpass
from datetime import datetime

def ensure_output_directory():
    """Create output directory for visualizations if it doesn't exist."""
    output_dir = "embedding_visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def save_plot(plt, filename):
    """Save the current plot to the visualizations directory."""
    output_dir = ensure_output_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(output_dir, f"{filename}_{timestamp}.png")
    plt.savefig(full_path)
    print(f"Saved visualization to: {full_path}")
    plt.close()

def create_embedding(text, client):
    """Create an embedding for the given text."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    )
    return np.array(response.data[0].embedding)

def visualize_embedding_stats(embedding):
    """Visualize basic statistics about the embedding vector."""
    plt.figure(figsize=(12, 4))
    
    # Plot histogram of values
    plt.subplot(131)
    plt.hist(embedding, bins=50)
    plt.title('Distribution of Vector Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # Plot first 50 dimensions
    plt.subplot(132)
    plt.plot(embedding[:50])
    plt.title('First 50 Dimensions')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    
    # Basic statistics
    stats = f"""
    Mean: {np.mean(embedding):.4f}
    Std: {np.std(embedding):.4f}
    Min: {np.min(embedding):.4f}
    Max: {np.max(embedding):.4f}
    """
    plt.subplot(133)
    plt.text(0.1, 0.5, stats, fontsize=10)
    plt.axis('off')
    plt.title('Vector Statistics')
    
    plt.tight_layout()
    save_plot(plt, "embedding_stats")

def compare_similar_texts(client):
    """Compare embeddings of similar but different texts."""
    texts = [
        "What is the capital of France?",
        "Tell me France's capital city",
        "Paris is located in which country?",
        "What is the capital of Germany?"  # Different meaning
    ]
    
    # Create embeddings for all texts
    embeddings = [create_embedding(text, client) for text in texts]
    
    # Calculate cosine similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    similarities = []
    for i in range(len(embeddings)):
        row = []
        for j in range(len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            row.append(f"{sim:.3f}")
        similarities.append(row)
    
    # Visualize similarities
    plt.figure(figsize=(10, 8))
    plt.imshow([[float(x) for x in row] for row in similarities], cmap='YlOrRd')
    plt.colorbar()
    
    # Add text annotations
    for i in range(len(texts)):
        for j in range(len(texts)):
            plt.text(j, i, similarities[i][j], ha='center', va='center')
    
    plt.xticks(range(len(texts)), [textwrap.fill(t, 15) for t in texts], rotation=45)
    plt.yticks(range(len(texts)), [textwrap.fill(t, 15) for t in texts])
    plt.title('Cosine Similarity Between Different Prompts')
    plt.tight_layout()
    save_plot(plt, "similarity_matrix")

def get_api_key():
    """Prompt for OpenAI API key with secure input."""
    print("\nPlease enter your OpenAI API key.")
    print("Note: The input will be hidden for security.")
    api_key = getpass("API Key: ")
    return api_key

def main():
    # Get API key from user
    api_key = get_api_key()
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Original prompt
        text_prompt = "What is the capital of France?"
        print(f"\nCreating embedding for: '{text_prompt}'")
        
        # Create and analyze the embedding
        embedding = create_embedding(text_prompt, client)
        
        print(f"\nEmbedding shape: {embedding.shape}")
        print(f"Number of dimensions: {len(embedding)}")
        print("\nFirst 10 dimensions of the embedding vector:")
        print(embedding[:10])
        
        # Visualize the embedding
        print("\nVisualizing embedding statistics...")
        visualize_embedding_stats(embedding)
        
        # Compare similar texts
        print("\nComparing similar texts...")
        compare_similar_texts(client)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    main()