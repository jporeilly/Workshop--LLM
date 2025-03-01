import numpy as np  # For numerical operations and array handling
from typing import List, Dict, Tuple  # Type hints for better code documentation
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg for non-interactive environments (e.g., servers)
import matplotlib.pyplot as plt  # For creating visualizations
from sklearn.metrics.pairwise import cosine_similarity  # For calculating similarity between vectors
from sklearn.manifold import TSNE  # For dimensionality reduction to visualize high-dimensional data
import seaborn as sns  # For enhanced visualizations on top of matplotlib
import pandas as pd  # For data manipulation and analysis
import os  # For file and directory operations
from datetime import datetime  # For timestamping output files
import ollama  # Python client for interacting with Ollama API

class EmbeddingAnalyzer:
    """
    A class to analyze and visualize text embeddings using Ollama.
    
    This class provides methods to:
    - Generate embeddings for text using Ollama's llama3.2 model
    - Calculate similarities between texts
    - Visualize embedding properties and relationships
    - Create semantic search demonstrations
    """
    def __init__(self, output_dir: str, host: str = "http://localhost:11434"):
        """
        Initialize the analyzer with Ollama client and output directory.
        
        Args:
            output_dir: Directory to save visualizations and analysis results
            host: Ollama server host URL (default: http://localhost:11434)
        """
        # Initialize the Ollama client with the specified host
        self.client = ollama.Client(host=host)
        # Specify which Ollama model to use for embeddings
        self.model = "llama3.2:latest"
        # Cache to store embeddings to avoid regenerating for the same text
        self.cache: Dict[str, np.ndarray] = {}
        # Directory where all output files will be saved
        self.output_dir = output_dir
        
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for the input text, using cache if available.
        
        An embedding is a numerical representation of text in a high-dimensional space,
        where semantic meaning is captured by the relative positions of vectors.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            A numpy array containing the embedding vector
        """
        # Check if embedding is already in cache to avoid redundant API calls
        if text in self.cache:
            return self.cache[text]
        
        # Request embedding from Ollama API    
        response = self.client.embeddings(
            model=self.model,  # Using the specified Ollama model
            prompt=text  # The text to embed
        )
        
        # Convert the embedding to numpy array for easier manipulation
        embedding = np.array(response["embedding"])
        
        # Store in cache for future use
        self.cache[text] = embedding
        
        return embedding
    
    def batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of numpy arrays, each containing an embedding vector
        """
        # Generate embeddings for each text in the list
        return [self.get_embedding(text) for text in texts]
    
    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Calculate pairwise similarities between all provided texts.
        
        This creates a matrix where each cell [i,j] contains the cosine similarity
        between the embeddings of texts[i] and texts[j].
        
        Args:
            texts: List of text strings to compare
            
        Returns:
            A 2D numpy array containing pairwise similarity scores
        """
        # Get embeddings for all texts
        embeddings = self.batch_embed(texts)
        
        # Stack vectors vertically to create a 2D matrix
        # Each row is an embedding vector for one text
        embeddings_matrix = np.vstack(embeddings)
        
        # Calculate cosine similarity between all pairs of vectors
        # Output is a square matrix of size len(texts) × len(texts)
        return cosine_similarity(embeddings_matrix)
    
    def save_plot(self, plt, filename: str) -> str:
        """
        Save plot to the output directory.
        
        Args:
            plt: Matplotlib plot object to save
            filename: Name of the file to save the plot as
            
        Returns:
            Full path to the saved file
        """
        # Create full path for the output file
        full_path = os.path.join(self.output_dir, filename)
        
        # Save the figure to the specified path
        plt.savefig(full_path)
        
        # Close the plot to free memory
        plt.close()
        
        print(f"Saved visualization to: {full_path}")
        return full_path
    
    def visualize_similarities(self, texts: List[str], labels: List[str] = None, filename: str = 'similarity_heatmap.png'):
        """
        Create a heatmap visualization of text similarities and save to file.
        
        Args:
            texts: List of text strings to compare
            labels: Optional labels for each text (default: numbered indices)
            filename: Name of the output file
        """
        # Calculate the similarity matrix for all texts
        similarity_matrix = self.calculate_similarity_matrix(texts)
        
        # Create figure with appropriate size
        plt.figure(figsize=(10, 8))
        
        # Create heatmap using seaborn
        sns.heatmap(
            similarity_matrix,
            annot=True,  # Show the similarity values in each cell
            fmt='.2f',   # Format as 2 decimal places
            cmap='YlOrRd',  # Color map: yellow to orange to red (higher values are redder)
            xticklabels=labels or range(len(texts)),  # Use provided labels or default to indices
            yticklabels=labels or range(len(texts))
        )
        
        # Add title and adjust layout
        plt.title('Semantic Similarity Heatmap')
        plt.tight_layout()
        
        # Save the visualization
        self.save_plot(plt, filename)
    
    def visualize_embedding_clusters(self, texts: List[str], labels: List[str] = None, filename: str = 'embedding_clusters.png'):
        """
        Create a 2D visualization of embedding clusters using t-SNE dimensionality reduction.
        
        This visualizes how different texts relate to each other in the embedding space
        by projecting the high-dimensional embeddings down to 2D.
        
        Args:
            texts: List of text strings to visualize
            labels: Optional category labels for each text
            filename: Name of the output file
        """
        # Get embeddings for all texts
        embeddings = self.batch_embed(texts)
        
        # Stack vectors vertically to create a 2D matrix
        embeddings_matrix = np.vstack(embeddings)
        
        # Calculate appropriate perplexity for t-SNE
        # Perplexity is related to the number of nearest neighbors used in the algorithm
        # It should be smaller than the number of points - 1
        n_samples = len(texts)
        perplexity = min(30, n_samples - 1)
        
        # Create t-SNE model for dimensionality reduction
        # t-SNE (t-Distributed Stochastic Neighbor Embedding) preserves local relationships
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        
        # Transform the high-dimensional embeddings to 2D points
        reduced_embeddings = tsne.fit_transform(embeddings_matrix)
        
        # Create DataFrame for easier plotting with seaborn
        df = pd.DataFrame(
            reduced_embeddings,
            columns=['x', 'y']  # 2D coordinates
        )
        # Add labels column for coloring points by category
        df['label'] = labels if labels else range(len(texts))
        
        # Create figure with appropriate size
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot using seaborn
        # Points with the same label will have the same color and marker style
        sns.scatterplot(data=df, x='x', y='y', hue='label', style='label')
        
        # Add title and adjust layout
        plt.title('2D Visualization of Text Embeddings')
        plt.tight_layout()
        
        # Save the visualization
        self.save_plot(plt, filename)

def ensure_output_directory() -> str:
    """
    Create and return the output directory path with timestamp.
    
    Creates a unique directory for each run of the script to prevent
    overwriting previous results.
    
    Returns:
        Full path to the created output directory
    """
    # Base directory for all analysis outputs
    base_dir = "embedding_analysis"
    
    # Generate timestamp for unique directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create full path with timestamp
    output_dir = os.path.join(base_dir, f"analysis_{timestamp}")
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir

def get_ollama_host() -> str:
    """
    Prompt for Ollama host URL with default option.
    
    Allows connecting to either the default local Ollama server
    or a custom server specified by the user.
    
    Returns:
        Host URL for the Ollama API
    """
    # Default local Ollama server URL
    default_host = "http://localhost:11434"
    
    print("\nOllama Configuration")
    print("===================")
    print(f"Default Ollama server: {default_host}")
    
    # Ask if user wants to use a different server
    use_custom = input("Use a different Ollama server? (y/N): ").lower()
    
    if use_custom in ('y', 'yes'):
        # Get custom host URL
        host = input(f"Enter Ollama server URL: ")
        # Use provided URL or fall back to default if empty
        return host if host else default_host
    
    return default_host

def save_analysis_results(output_dir: str, results: str):
    """
    Save analysis results to a text file.
    
    Args:
        output_dir: Directory to save the file in
        results: Text content to save
    """
    # Create full path for the output file
    filename = os.path.join(output_dir, "analysis_results.txt")
    
    # Write results to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(results)
    
    print(f"Analysis results saved to: {filename}")

def demonstrate_embeddings():
    """
    Demonstrate various applications and properties of embeddings.
    
    This function showcases different ways embeddings can be used:
    1. Measuring semantic similarity between texts
    2. Clustering texts by topic
    3. Analyzing embedding vector properties
    4. Performing semantic search
    """
    # Create output directory for this run
    output_dir = ensure_output_directory()
    print(f"\nAnalysis results will be saved to: {output_dir}")
    
    # Get Ollama host configuration
    host = get_ollama_host()
    
    try:
        # Initialize analyzer with Ollama
        print(f"\nInitializing EmbeddingAnalyzer with Ollama (model: llama3.2:latest)")
        analyzer = EmbeddingAnalyzer(output_dir, host)
        
        # Example 1: Basic Semantic Similarity
        # This demonstrates how embeddings capture semantic relationships
        print("\nExample 1: Basic Semantic Similarity")
        similar_texts = [
            "What is the capital of France?",
            "Tell me the capital city of France",
            "Which city serves as France's capital?",
            "What's the largest city in France?",
            "What's the weather like in Paris?"
        ]
        # Create heatmap of similarities between these related texts
        analyzer.visualize_similarities(
            similar_texts, 
            labels=[f"Text {i+1}" for i in range(len(similar_texts))],
            filename="similarity_heatmap.png"
        )
        
        # Example 2: Topic Clustering
        # This demonstrates how embeddings group semantically related concepts
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
        # Create labels for each topic category
        topic_labels = ["Tech"]*3 + ["Sports"]*3 + ["Cooking"]*3
        
        # Visualize how these topics cluster in the embedding space
        analyzer.visualize_embedding_clusters(
            mixed_topics, 
            labels=topic_labels,
            filename="embedding_clusters.png"
        )
        
        # Example 3: Embedding Properties Analysis
        # This demonstrates the statistical properties of embedding vectors
        print("\nExample 3: Analyzing Embedding Properties")
        sample_text = "This is a sample text for analyzing embedding properties."
        embedding = analyzer.get_embedding(sample_text)
        
        # Create histogram of embedding values
        plt.figure(figsize=(10, 5))
        plt.hist(embedding, bins=50)
        plt.title("Distribution of Embedding Values")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        analyzer.save_plot(plt, 'embedding_distribution.png')
        
        # Collect statistical properties of the embedding
        stats = f"""Embedding Analysis Results
        -------------------------
        Sample Text: "{sample_text}"
        Model: {analyzer.model}
        
        Embedding Statistics:
        - Dimensionality: {len(embedding)} dimensions
        - Mean value: {np.mean(embedding):.4f}
        - Standard deviation: {np.std(embedding):.4f}
        - Vector magnitude: {np.linalg.norm(embedding):.4f}
        """
        
        # Example 4: Semantic Search
        # This demonstrates using embeddings for finding similar documents
        print("\nExample 4: Semantic Search Demo")
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "A fast auburn canine leaps across a sleepy hound",
            "The cat chases the mouse in the garden",
            "A feline pursues a rodent through the flowers",
            "The weather is sunny and warm today",
        ]
        # Query to search for
        query = "A fox jumping over a dog"
        query_embedding = analyzer.get_embedding(query)
        
        # Calculate similarity scores between query and all documents
        doc_embeddings = analyzer.batch_embed(documents)
        similarities = [
            cosine_similarity(query_embedding.reshape(1, -1), doc_emb.reshape(1, -1))[0][0]
            for doc_emb in doc_embeddings
        ]
        
        # Add search results to stats
        stats += "\nSemantic Search Results:\n"
        stats += f"Query: '{query}'\n\n"
        
        # Sort documents by similarity score (highest first)
        for doc, score in sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True):
            stats += f"Score: {score:.4f} | Document: {doc}\n"
        
        # Save all analysis results to text file
        save_analysis_results(output_dir, stats)
        
        print("\nAnalysis complete! All visualizations and results have been saved.")
        
    except Exception as e:
        # Handle errors with helpful troubleshooting information
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Ensure Ollama is installed and running (see https://ollama.com)")
        print("2. Check if the llama3.2:latest model is pulled (`ollama pull llama3.2:latest`)")
        print("3. Verify the Ollama server URL is correct")
        print("4. Make sure the ollama Python package is installed (`pip install ollama`)")
        print(f"\nError details: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    # Entry point of the script
    # This ensures the script only runs when executed directly, not when imported
    demonstrate_embeddings()