from openai import OpenAI
import numpy as np
from typing import List, Dict, Tuple
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for non-interactive environments
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
from getpass import getpass

class EmbeddingAnalyzer:
    """
    A class to analyze and visualize text embeddings.
    """
    def __init__(self, api_key: str, output_dir: str):
        """
        Initialize the analyzer with OpenAI credentials and output directory.
        
        Args:
            api_key: OpenAI API key
            output_dir: Directory to save visualizations and analysis
        """
        self.client = OpenAI(api_key=api_key)
        self.cache: Dict[str, np.ndarray] = {}  # Cache for embeddings
        self.output_dir = output_dir
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding vector for the input text, using cache if available."""
        if text in self.cache:
            return self.cache[text]
            
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        embedding = np.array(response.data[0].embedding)
        self.cache[text] = embedding
        return embedding
    
    def batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [self.get_embedding(text) for text in texts]
    
    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Calculate pairwise similarities between all provided texts."""
        embeddings = self.batch_embed(texts)
        embeddings_matrix = np.vstack(embeddings)
        return cosine_similarity(embeddings_matrix)
    
    def save_plot(self, plt, filename: str) -> str:
        """Save plot to the output directory with timestamp."""
        full_path = os.path.join(self.output_dir, filename)
        plt.savefig(full_path)
        plt.close()
        print(f"Saved visualization to: {full_path}")
        return full_path
    
    def visualize_similarities(self, texts: List[str], labels: List[str] = None, filename: str = 'similarity_heatmap.png'):
        """Create a heatmap visualization of text similarities and save to file."""
        similarity_matrix = self.calculate_similarity_matrix(texts)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            xticklabels=labels or range(len(texts)),
            yticklabels=labels or range(len(texts))
        )
        plt.title('Semantic Similarity Heatmap')
        plt.tight_layout()
        self.save_plot(plt, filename)
    
    def visualize_embedding_clusters(self, texts: List[str], labels: List[str] = None, filename: str = 'embedding_clusters.png'):
        """Create a 2D visualization of embedding clusters using t-SNE."""
        embeddings = self.batch_embed(texts)
        embeddings_matrix = np.vstack(embeddings)
        
        # Calculate appropriate perplexity
        n_samples = len(texts)
        perplexity = min(30, n_samples - 1)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        reduced_embeddings = tsne.fit_transform(embeddings_matrix)
        
        df = pd.DataFrame(
            reduced_embeddings,
            columns=['x', 'y']
        )
        df['label'] = labels if labels else range(len(texts))
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='x', y='y', hue='label', style='label')
        plt.title('2D Visualization of Text Embeddings')
        plt.tight_layout()
        self.save_plot(plt, filename)

def ensure_output_directory() -> str:
    """Create and return the output directory path with timestamp."""
    base_dir = "embedding_analysis"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"analysis_{timestamp}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def get_api_key() -> str:
    """Prompt for OpenAI API key with secure input."""
    print("\nPlease enter your OpenAI API key.")
    print("Note: The input will be hidden for security.")
    api_key = getpass("API Key: ")
    return api_key

def save_analysis_results(output_dir: str, results: str):
    """Save analysis results to a text file."""
    filename = os.path.join(output_dir, "analysis_results.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(results)
    print(f"Analysis results saved to: {filename}")

def demonstrate_embeddings():
    """Demonstrate various applications and properties of embeddings."""
    # Create output directory and get API key
    output_dir = ensure_output_directory()
    print(f"\nAnalysis results will be saved to: {output_dir}")
    
    api_key = get_api_key()
    
    try:
        # Initialize analyzer
        analyzer = EmbeddingAnalyzer(api_key, output_dir)
        
        # Example 1: Basic Semantic Similarity
        print("\nExample 1: Basic Semantic Similarity")
        similar_texts = [
            "What is the capital of France?",
            "Tell me the capital city of France",
            "Which city serves as France's capital?",
            "What's the largest city in France?",
            "What's the weather like in Paris?"
        ]
        analyzer.visualize_similarities(
            similar_texts, 
            labels=[f"Text {i+1}" for i in range(len(similar_texts))],
            filename="similarity_heatmap.png"
        )
        
        # Example 2: Topic Clustering
        print("\nExample 2: Topic Clustering")
        mixed_topics = [
            # Technology
            "How do computers process information?",
            "What is artificial intelligence?",
            "How does machine learning work?",
            # Sports
            "Who won the last World Cup?",
            "What are the rules of basketball?",
            "How do you play tennis?",
            # Cooking
            "What's the best way to cook pasta?",
            "How do you make chocolate cake?",
            "What are common cooking spices?"
        ]
        topic_labels = ["Tech"]*3 + ["Sports"]*3 + ["Cooking"]*3
        analyzer.visualize_embedding_clusters(
            mixed_topics, 
            labels=topic_labels,
            filename="embedding_clusters.png"
        )
        
        # Example 3: Embedding Properties Analysis
        print("\nExample 3: Analyzing Embedding Properties")
        sample_text = "This is a sample text for analyzing embedding properties."
        embedding = analyzer.get_embedding(sample_text)
        
        plt.figure(figsize=(10, 5))
        plt.hist(embedding, bins=50)
        plt.title("Distribution of Embedding Values")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        analyzer.save_plot(plt, 'embedding_distribution.png')
        
        # Collect statistical properties
        stats = f"""Embedding Analysis Results
        -------------------------
        Sample Text: "{sample_text}"
        
        Embedding Statistics:
        - Dimensionality: {len(embedding)} dimensions
        - Mean value: {np.mean(embedding):.4f}
        - Standard deviation: {np.std(embedding):.4f}
        - Vector magnitude: {np.linalg.norm(embedding):.4f}
        """
        
        # Example 4: Semantic Search
        print("\nExample 4: Semantic Search Demo")
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "A fast auburn canine leaps across a sleepy hound",
            "The cat chases the mouse in the garden",
            "A feline pursues a rodent through the flowers",
            "The weather is sunny and warm today",
        ]
        query = "A fox jumping over a dog"
        query_embedding = analyzer.get_embedding(query)
        
        # Calculate similarities with all documents
        doc_embeddings = analyzer.batch_embed(documents)
        similarities = [
            cosine_similarity(query_embedding.reshape(1, -1), doc_emb.reshape(1, -1))[0][0]
            for doc_emb in doc_embeddings
        ]
        
        # Add search results to stats
        stats += "\nSemantic Search Results:\n"
        stats += f"Query: '{query}'\n\n"
        for doc, score in sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True):
            stats += f"Score: {score:.4f} | Document: {doc}\n"
        
        # Save all analysis results
        save_analysis_results(output_dir, stats)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    demonstrate_embeddings()