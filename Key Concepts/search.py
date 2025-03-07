import numpy as np  # For numerical operations and array handling
from typing import List, Dict, Tuple  # Type hints for better code documentation
import matplotlib.pyplot as plt  # For creating visualizations
from sklearn.metrics.pairwise import cosine_similarity  # For calculating similarity between vectors
from sklearn.manifold import TSNE  # For dimensionality reduction to visualize high-dimensional data
import seaborn as sns  # For enhanced visualizations on top of matplotlib
import pandas as pd  # For data manipulation and analysis
from collections import Counter  # For counting word frequencies in keyword search
import re  # For regular expressions to extract words
import os  # For file and directory operations
from datetime import datetime  # For timestamping output files
import ollama  # Python client for interacting with Ollama API

def ensure_output_directory() -> str:
    """
    Create and return the output directory path with timestamp.
    
    This function creates a unique directory for each run of the script
    to prevent overwriting previous results.
    
    Returns:
        str: Path to the created output directory
    """
    # Base directory for search analysis outputs
    base_dir = "search_analysis"
    
    # Generate timestamp for unique directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create full path with timestamp
    output_dir = os.path.join(base_dir, f"analysis_{timestamp}")
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    return output_dir

def get_ollama_host() -> str:
    """
    Prompt for Ollama host URL with default option.
    
    This function allows the user to specify a custom Ollama server
    or use the default localhost URL.
    
    Returns:
        str: Host URL for the Ollama API
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
        # Return provided URL or fall back to default if empty
        return host if host else default_host
    
    return default_host

class SearchComparator:
    """
    A class to compare traditional keyword search with embedding-based semantic search.
    
    This class provides methods to:
    - Generate embeddings using Ollama's llama3.2:latest model
    - Perform keyword-based search using term frequency
    - Perform vector-based semantic search using embeddings
    - Visualize and compare results from both search methods
    """
    
    def __init__(self, ollama_host: str, output_dir: str):
        """
        Initialize with Ollama host and output directory.
        
        Args:
            ollama_host: URL of the Ollama API server
            output_dir: Directory to save visualizations and analysis results
        """
        # Initialize the Ollama client with the specified host
        self.client = ollama.Client(host=ollama_host)
        
        # Specify which Ollama model to use for embeddings
        self.model = "llama3.2:latest"
        
        # Cache to store embeddings to avoid regenerating for the same text
        self.cache: Dict[str, np.ndarray] = {}
        
        # Directory where all output files will be saved
        self.output_dir = output_dir
        
    def get_search_type(self, query: str) -> str:
        """
        Determine the type of search based on the query.
        
        This helps categorize different types of searches for analysis and
        provides appropriate naming for output files.
        
        Args:
            query: The search query string
            
        Returns:
            str: A category name for the search type
        """
        # Map queries to search types for analysis and file naming
        search_types = {
            "A fox jumping over a dog": "direct_phrase_match",
            "Canines in natural habitats": "semantic_concept_match",
            "Sleeping animals outdoors": "mixed_concept_match",
            "Forest wildlife activity": "thematic_match"
        }
        # Return the mapped type or "custom_search" if not in the predefined list
        return search_types.get(query, "custom_search")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding vector for the input text using Ollama.
        
        This function uses caching to avoid redundant API calls for the same text.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        # Check if embedding is already in cache to avoid redundant API calls
        if text in self.cache:
            return self.cache[text]
        
        # Request embedding from Ollama API
        response = self.client.embeddings(
            model=self.model,  # Using llama3.2:latest model
            prompt=text  # The text to embed
        )
        
        # Convert the embedding to numpy array for easier manipulation
        embedding = np.array(response["embedding"])
        
        # Store in cache for future use
        self.cache[text] = embedding
        
        return embedding
    
    def keyword_search(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """
        Perform traditional keyword-based search using term frequency.
        
        This simulates a simple TF (Term Frequency) based search by counting
        how many times each query word appears in each document.
        
        Args:
            query: The search query string
            documents: List of document strings to search
            
        Returns:
            List of (document, score) tuples, sorted by score in descending order
        """
        # Extract lowercase tokens (words) from the query
        query_tokens = set(re.findall(r'\w+', query.lower()))
        
        results = []
        for doc in documents:
            # Count frequency of all words in the document
            doc_tokens = Counter(re.findall(r'\w+', doc.lower()))
            
            # Score is the sum of frequencies of query words that appear in the document
            score = sum(doc_tokens[token] for token in query_tokens if token in doc_tokens)
            
            # Add document and its score to results
            results.append((doc, score))
        
        # Sort results by score in descending order (highest first)
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def vector_search(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """
        Perform vector-based semantic search using embeddings.
        
        This uses cosine similarity between the query embedding and each document
        embedding to find semantically similar documents.
        
        Args:
            query: The search query string
            documents: List of document strings to search
            
        Returns:
            List of (document, similarity_score) tuples, sorted by score in descending order
        """
        # Get embedding for the query
        query_embedding = self.get_embedding(query)
        results = []
        
        for doc in documents:
            # Get embedding for the document
            doc_embedding = self.get_embedding(doc)
            
            # Calculate cosine similarity between query and document embeddings
            # Reshape is needed because cosine_similarity expects 2D arrays
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1), 
                doc_embedding.reshape(1, -1)
            )[0][0]
            
            # Add document and its similarity score to results
            results.append((doc, similarity))
        
        # Sort results by similarity score in descending order (highest first)
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def save_visualization(self, fig, search_type: str, viz_type: str) -> str:
        """
        Save visualization with appropriate naming.
        
        Args:
            fig: Matplotlib figure to save
            search_type: Category of the search (e.g., "direct_phrase_match")
            viz_type: Type of visualization (e.g., "comparison")
            
        Returns:
            str: Path to the saved file
        """
        # Create filename using search type and visualization type
        filename = f"{search_type}_{viz_type}.png"
        
        # Create full filepath in the output directory
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the figure
        fig.savefig(filepath)
        
        # Close the figure to free memory
        plt.close(fig)
        
        return filepath
    
    def print_and_save_results(self, query: str, keyword_results: List[Tuple[str, float]], 
                             vector_results: List[Tuple[str, float]], search_type: str):
        """
        Print results to console and save to file.
        
        This function displays the top results from both search methods and
        saves the complete results to a text file.
        
        Args:
            query: The search query string
            keyword_results: Results from keyword search
            vector_results: Results from vector search
            search_type: Category of the search (for filename)
        """
        # Print to console
        print(f"\nAnalyzing search results for query: '{query}'")
        
        # Show top 3 keyword search results
        print("\nKeyword Search Results:")
        for doc, score in keyword_results[:3]:
            print(f"Score: {score:.4f} | {doc}")
        
        # Show top 3 vector search results
        print("\nVector Search Results:")
        for doc, score in vector_results[:3]:
            print(f"Score: {score:.4f} | {doc}")
        
        # Create filename for results text file
        filename = f"{search_type}_results.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save complete results to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Search Results Analysis for Query: '{query}'\n")
            f.write("=" * 50 + "\n\n")
            
            # Write all keyword search results
            f.write("Keyword Search Results:\n")
            f.write("-" * 20 + "\n")
            for doc, score in keyword_results:
                f.write(f"Score: {score:.4f} | {doc}\n")
            
            # Write all vector search results
            f.write("\nVector Search Results:\n")
            f.write("-" * 20 + "\n")
            for doc, score in vector_results:
                f.write(f"Score: {score:.4f} | {doc}\n")
            
            # Add model information
            f.write("\n\nEmbedding Model: Ollama - " + self.model + "\n")
    
    def visualize_search_comparison(self, query: str, documents: List[str]):
        """
        Create visualizations comparing keyword and vector search results.
        
        This function runs both search methods and generates visualizations
        to compare their results.
        
        Args:
            query: The search query string
            documents: List of document strings to search
        """
        # Determine the type of search for categorization and file naming
        search_type = self.get_search_type(query)
        
        # Get search results from both methods
        keyword_results = self.keyword_search(query, documents)
        vector_results = self.vector_search(query, documents)
        
        # Print to console and save to text file
        self.print_and_save_results(query, keyword_results, vector_results, search_type)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # Create and save bar chart comparison
        fig1 = self.create_comparison_plot(keyword_results, vector_results, documents)
        comparison_path = self.save_visualization(fig1, search_type, "comparison")
        
        # Create and save embedding space visualization
        fig2 = self.visualize_query_document_space(query, documents)
        embedding_path = self.save_visualization(fig2, search_type, "embedding_space")
        
        print(f"Visualizations saved as '{os.path.basename(comparison_path)}' and '{os.path.basename(embedding_path)}'")
    
    def create_comparison_plot(self, keyword_results: List[Tuple[str, float]], 
                             vector_results: List[Tuple[str, float]], 
                             documents: List[str]) -> plt.Figure:
        """
        Create comparison plot of keyword and vector search results.
        
        This generates a side-by-side bar chart comparing the scores from
        both search methods.
        
        Args:
            keyword_results: Results from keyword search
            vector_results: Results from vector search
            documents: List of document strings (for ordering)
            
        Returns:
            matplotlib.pyplot.Figure: The generated figure
        """
        # Extract scores from both search results
        # The results are already sorted by score, so we need to match with original document order
        doc_to_keyword = {doc: score for doc, score in keyword_results}
        doc_to_vector = {doc: score for doc, score in vector_results}
        
        # Get scores in document order
        keyword_scores = [doc_to_keyword.get(doc, 0) for doc in documents]
        vector_scores = [doc_to_vector.get(doc, 0) for doc in documents]
        
        # Normalize keyword scores for better comparison with similarity scores
        max_keyword = max(keyword_scores) if max(keyword_scores) > 0 else 1
        keyword_scores = [s/max_keyword for s in keyword_scores]
        
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Keyword search results - left subplot
        bars1 = ax1.bar(range(len(documents)), keyword_scores, alpha=0.6)
        ax1.set_title('Keyword Search Results')
        ax1.set_xlabel('Document Index')
        ax1.set_ylabel('Normalized Score')
        ax1.set_xticks(range(len(documents)))
        ax1.set_xticklabels([f'Doc {i}' for i in range(len(documents))], rotation=45)
        
        # Add score labels on top of each bar
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Vector search results - right subplot
        bars2 = ax2.bar(range(len(documents)), vector_scores, alpha=0.6)
        ax2.set_title('Vector Search Results')
        ax2.set_xlabel('Document Index')
        ax2.set_ylabel('Similarity Score')
        ax2.set_xticks(range(len(documents)))
        ax2.set_xticklabels([f'Doc {i}' for i in range(len(documents))], rotation=45)
        
        # Add score labels on top of each bar
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Add overall title for the figure
        plt.suptitle('Comparison of Search Methods', fontsize=16)
        plt.tight_layout()
        return fig
    
    def visualize_query_document_space(self, query: str, documents: List[str]) -> plt.Figure:
        """
        Create a 2D visualization of query and documents in embedding space.
        
        This uses t-SNE to reduce the high-dimensional embeddings to 2D for visualization,
        showing how the query relates to documents in semantic space.
        
        Args:
            query: The search query string
            documents: List of document strings
            
        Returns:
            matplotlib.pyplot.Figure: The generated figure
        """
        # Combine query and documents into a single list
        all_texts = [query] + documents
        
        # Get embeddings for all texts
        print("Generating embeddings for visualization...")
        embeddings = [self.get_embedding(text) for text in all_texts]
        
        # Stack vectors vertically to create a 2D matrix
        embeddings_matrix = np.vstack(embeddings)
        
        # Calculate appropriate perplexity for t-SNE
        # Perplexity is related to number of nearest neighbors considered
        # It should be smaller than the number of points - 1
        n_samples = len(all_texts)
        perplexity = min(30, n_samples - 1)
        
        # Reduce dimensionality with t-SNE
        print("Reducing dimensionality with t-SNE...")
        tsne = TSNE(
            n_components=2,  # Reduce to 2D for visualization
            random_state=42,  # For reproducibility
            perplexity=perplexity,
            max_iter=1000  # More iterations for better convergence
        )
        reduced_embeddings = tsne.fit_transform(embeddings_matrix)
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame(
            reduced_embeddings,
            columns=['x', 'y']  # 2D coordinates
        )
        # Add type column to distinguish query from documents
        df['type'] = ['Query'] + ['Document'] * len(documents)
        # Add the original text
        df['text'] = all_texts
        
        # Create visualization
        fig = plt.figure(figsize=(12, 8))
        
        # Create scatter plot with seaborn
        sns.scatterplot(
            data=df, 
            x='x', 
            y='y', 
            hue='type',  # Color by type (Query vs Document)
            style='type',  # Different marker styles for Query vs Document
            s=100,  # Marker size
            palette={'Query': 'red', 'Document': 'blue'}  # Color palette
        )
        
        # Add text labels to the points
        for idx, row in df.iterrows():
            text = f"Query" if idx == 0 else f"Doc {idx-1}"
            plt.annotate(
                text,  # The label text
                (row['x'], row['y']),  # Point to label
                xytext=(5, 5),  # Offset text position
                textcoords='offset points',  # How to interpret the offset
                # Add white background to text for better readability
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
        
        # Add descriptive title
        plt.title('2D Visualization of Query and Documents in Embedding Space')
        plt.tight_layout()
        return fig

def demonstrate_search_comparison():
    """
    Demonstrate the differences between keyword and semantic search.
    
    This function:
    1. Sets up the environment (output directory and Ollama connection)
    2. Initializes the SearchComparator
    3. Runs comparisons on several test queries
    4. Generates visualizations for each comparison
    """
    print("Search Comparison Demo: Keyword vs. Vector Search using Ollama")
    print("=" * 65)
    print("This script compares traditional keyword search with embedding-based")
    print("semantic search using the llama3.2:latest model via Ollama.")
    
    try:
        # Create output directory
        output_dir = ensure_output_directory()
        print(f"\nResults will be saved to: {output_dir}")
        
        # Get Ollama host configuration
        ollama_host = get_ollama_host()
        
        # Initialize comparator
        print(f"\nInitializing SearchComparator with Ollama (model: llama3.2:latest)")
        comparator = SearchComparator(ollama_host, output_dir)
        
        # Test documents
        print("\nPreparing test documents...")
        documents = [
            "The rapid brown fox jumps over the lazy dog in the forest",
            "A quick auburn canine leaps across a sleepy hound in the woods",
            "The fox hunts for food in the dense woodland",
            "Dogs and other canines play together in the park",
            "A lazy afternoon in the garden with sleeping pets",
            "Wild animals roaming through the forest at night",
            "The weather is perfect for outdoor activities today",
            "Forest creatures gather near the stream at dusk"
        ]
        
        # Display the test documents
        print("\nTest Documents:")
        for i, doc in enumerate(documents):
            print(f"Doc {i}: {doc}")
        
        # Test queries
        queries = [
            "A fox jumping over a dog",           # Direct phrase match
            "Canines in natural habitats",        # Semantic concept match
            "Sleeping animals outdoors",          # Mixed concept match
            "Forest wildlife activity"            # Thematic match
        ]
        
        # Run comparisons for each query
        print("\nRunning search comparisons...")
        for query in queries:
            print(f"\n{'-' * 40}")
            print(f"Processing query: '{query}'")
            comparator.visualize_search_comparison(query, documents)
        
        print(f"\nAll comparisons complete! Results saved to {output_dir}")
            
    except Exception as e:
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
    demonstrate_search_comparison()