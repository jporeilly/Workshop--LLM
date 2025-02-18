from openai import OpenAI
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

class EmbeddingAnalyzer:
    """
    A class to analyze and visualize text embeddings.
    """
    def __init__(self, api_key: str):
        """Initialize the analyzer with OpenAI credentials."""
        self.client = OpenAI(api_key=api_key)
        self.cache: Dict[str, np.ndarray] = {}  # Cache for embeddings
        
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for the input text, using cache if available.
        
        Args:
            text: Input text to be embedded
            
        Returns:
            numpy array containing the embedding vector
        """
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
        """
        Calculate pairwise similarities between all provided texts.
        
        Args:
            texts: List of texts to compare
            
        Returns:
            2D numpy array of similarity scores
        """
        embeddings = self.batch_embed(texts)
        embeddings_matrix = np.vstack(embeddings)
        return cosine_similarity(embeddings_matrix)
    
    def visualize_similarities(self, texts: List[str], labels: List[str] = None):
        """
        Create a heatmap visualization of text similarities.
        
        Args:
            texts: List of texts to compare
            labels: Optional list of labels for the texts
        """
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
        plt.show()
    
    def visualize_embedding_clusters(self, texts: List[str], labels: List[str] = None):
        """
        Create a 2D visualization of embedding clusters using t-SNE.
        
        Args:
            texts: List of texts to visualize
            labels: Optional list of category labels for the texts
        """
        embeddings = self.batch_embed(texts)
        embeddings_matrix = np.vstack(embeddings)
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings_matrix)
        
        # Create DataFrame for plotting
        df = pd.DataFrame(
            reduced_embeddings,
            columns=['x', 'y']
        )
        df['label'] = labels if labels else range(len(texts))
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='x', y='y', hue='label', style='label')
        plt.title('2D Visualization of Text Embeddings')
        plt.show()

def demonstrate_embeddings():
    """
    Demonstrate various applications and properties of embeddings.
    """
    # Initialize analyzer (replace with your API key)
    analyzer = EmbeddingAnalyzer("YOUR_API_KEY")
    
    # Example 1: Basic Semantic Similarity
    print("\nExample 1: Basic Semantic Similarity")
    similar_texts = [
        "What is the capital of France?",
        "Tell me the capital city of France",
        "Which city serves as France's capital?",
        "What's the largest city in France?",
        "What's the weather like in Paris?"
    ]
    analyzer.visualize_similarities(similar_texts, labels=[f"Text {i+1}" for i in range(len(similar_texts))])
    
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
    analyzer.visualize_embedding_clusters(mixed_topics, labels=topic_labels)
    
    # Example 3: Embedding Properties Analysis
    print("\nExample 3: Analyzing Embedding Properties")
    sample_text = "This is a sample text for analyzing embedding properties."
    embedding = analyzer.get_embedding(sample_text)
    
    # Plot embedding value distribution
    plt.figure(figsize=(10, 5))
    plt.hist(embedding, bins=50)
    plt.title("Distribution of Embedding Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    
    # Print statistical properties
    print(f"\nEmbedding Statistics:")
    print(f"Dimensionality: {len(embedding)} dimensions")
    print(f"Mean value: {np.mean(embedding):.4f}")
    print(f"Standard deviation: {np.std(embedding):.4f}")
    print(f"Vector magnitude: {np.linalg.norm(embedding):.4f}")
    
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
    
    print("\nSemantic Search Results:")
    for doc, score in sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True):
        print(f"Score: {score:.4f} | Document: {doc}")

if __name__ == "__main__":
    demonstrate_embeddings()