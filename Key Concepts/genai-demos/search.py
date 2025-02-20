from openai import OpenAI
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from collections import Counter
import re
import os
from datetime import datetime
from getpass import getpass

def ensure_output_directory() -> str:
    """Create and return the output directory path with timestamp."""
    base_dir = "search_analysis"
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

class SearchComparator:
    def __init__(self, api_key: str, output_dir: str):
        """Initialize with API key and output directory."""
        self.client = OpenAI(api_key=api_key)
        self.cache: Dict[str, np.ndarray] = {}
        self.output_dir = output_dir
        
    def get_search_type(self, query: str) -> str:
        """Determine the type of search based on the query."""
        search_types = {
            "A fox jumping over a dog": "direct_phrase_match",
            "Canines in natural habitats": "semantic_concept_match",
            "Sleeping animals outdoors": "mixed_concept_match",
            "Forest wildlife activity": "thematic_match"
        }
        return search_types.get(query, "custom_search")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for input text."""
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
    
    def keyword_search(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Perform traditional keyword-based search using TF-IDF like scoring."""
        query_tokens = set(re.findall(r'\w+', query.lower()))
        
        results = []
        for doc in documents:
            doc_tokens = Counter(re.findall(r'\w+', doc.lower()))
            score = sum(doc_tokens[token] for token in query_tokens if token in doc_tokens)
            results.append((doc, score))
            
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def vector_search(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """Perform vector-based semantic search using embeddings."""
        query_embedding = self.get_embedding(query)
        results = []
        
        for doc in documents:
            doc_embedding = self.get_embedding(doc)
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1), 
                doc_embedding.reshape(1, -1)
            )[0][0]
            results.append((doc, similarity))
            
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def save_visualization(self, fig, search_type: str, viz_type: str) -> str:
        """Save visualization with appropriate naming."""
        filename = f"{search_type}_{viz_type}.png"
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)
        return filepath
    
    def print_and_save_results(self, query: str, keyword_results: List[Tuple[str, float]], 
                             vector_results: List[Tuple[str, float]], search_type: str):
        """Print results to console and save to file."""
        # Print to console
        print(f"\nAnalyzing search results for query: '{query}'")
        
        print("\nKeyword Search Results:")
        for doc, score in keyword_results[:3]:  # Show top 3 results
            print(f"Score: {score:.4f} | {doc}")
        
        print("\nVector Search Results:")
        for doc, score in vector_results[:3]:  # Show top 3 results
            print(f"Score: {score:.4f} | {doc}")
        
        # Save to file
        filename = f"{search_type}_results.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Search Results Analysis for Query: '{query}'\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Keyword Search Results:\n")
            f.write("-" * 20 + "\n")
            for doc, score in keyword_results:
                f.write(f"Score: {score:.4f} | {doc}\n")
            
            f.write("\nVector Search Results:\n")
            f.write("-" * 20 + "\n")
            for doc, score in vector_results:
                f.write(f"Score: {score:.4f} | {doc}\n")
    
    def visualize_search_comparison(self, query: str, documents: List[str]):
        """Create visualizations comparing keyword and vector search results."""
        search_type = self.get_search_type(query)
        
        # Get search results
        keyword_results = self.keyword_search(query, documents)
        vector_results = self.vector_search(query, documents)
        
        # Print and save results
        self.print_and_save_results(query, keyword_results, vector_results, search_type)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        
        # Create and save comparison plot
        fig1 = self.create_comparison_plot(keyword_results, vector_results, documents)
        comparison_path = self.save_visualization(fig1, search_type, "comparison")
        
        # Create and save embedding space visualization
        fig2 = self.visualize_query_document_space(query, documents)
        embedding_path = self.save_visualization(fig2, search_type, "embedding_space")
        
        print(f"Visualizations saved as '{os.path.basename(comparison_path)}' and '{os.path.basename(embedding_path)}'")
    
    def create_comparison_plot(self, keyword_results: List[Tuple[str, float]], 
                             vector_results: List[Tuple[str, float]], 
                             documents: List[str]) -> plt.Figure:
        """Create comparison plot of keyword and vector search results."""
        keyword_scores = [score for _, score in keyword_results]
        vector_scores = [score for _, score in vector_results]
        
        # Normalize keyword scores
        max_keyword = max(keyword_scores) if keyword_scores else 1
        keyword_scores = [s/max_keyword for s in keyword_scores]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Keyword search results
        bars1 = ax1.bar(range(len(documents)), keyword_scores, alpha=0.6)
        ax1.set_title('Keyword Search Results')
        ax1.set_xlabel('Document Index')
        ax1.set_ylabel('Normalized Score')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Vector search results
        bars2 = ax2.bar(range(len(documents)), vector_scores, alpha=0.6)
        ax2.set_title('Vector Search Results')
        ax2.set_xlabel('Document Index')
        ax2.set_ylabel('Similarity Score')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def visualize_query_document_space(self, query: str, documents: List[str]) -> plt.Figure:
        """Create a 2D visualization of query and documents in embedding space."""
        # Get embeddings
        all_texts = [query] + documents
        embeddings = [self.get_embedding(text) for text in all_texts]
        embeddings_matrix = np.vstack(embeddings)
        
        # Calculate appropriate perplexity
        n_samples = len(all_texts)
        perplexity = min(30, n_samples - 1)
        
        # Reduce dimensionality
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            max_iter=1000
        )
        reduced_embeddings = tsne.fit_transform(embeddings_matrix)
        
        # Create DataFrame
        df = pd.DataFrame(
            reduced_embeddings,
            columns=['x', 'y']
        )
        df['type'] = ['Query'] + ['Document'] * len(documents)
        df['text'] = all_texts
        
        # Create visualization
        fig = plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df, 
            x='x', 
            y='y', 
            hue='type',
            style='type',
            s=100,
            palette={'Query': 'red', 'Document': 'blue'}
        )
        
        # Add labels
        for idx, row in df.iterrows():
            text = f"Query" if idx == 0 else f"Doc {idx}"
            plt.annotate(
                text,
                (row['x'], row['y']),
                xytext=(5, 5),
                textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
            
        plt.title('2D Visualization of Query and Documents in Embedding Space')
        return fig

def demonstrate_search_comparison():
    """Demonstrate the differences between keyword and semantic search."""
    # Create output directory and get API key
    output_dir = ensure_output_directory()
    
    try:
        # Get API key securely
        api_key = get_api_key()
        
        # Initialize comparator
        comparator = SearchComparator(api_key, output_dir)
        
        # Test documents
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
        
        # Test queries
        queries = [
            "A fox jumping over a dog",  # Direct phrase match
            "Canines in natural habitats",  # Semantic concept match
            "Sleeping animals outdoors",  # Mixed concept match
            "Forest wildlife activity"  # Thematic match
        ]
        
        for query in queries:
            comparator.visualize_search_comparison(query, documents)
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    demonstrate_search_comparison()