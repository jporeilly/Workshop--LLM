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

class SearchComparator:
    """
    A class to compare different search methodologies: keyword-based vs semantic/vector search
    """
    def __init__(self, api_key: str):
        """Initialize with OpenAI credentials."""
        self.client = OpenAI(api_key=api_key)
        self.cache: Dict[str, np.ndarray] = {}
        
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
        """
        Perform traditional keyword-based search using TF-IDF like scoring.
        
        Returns: List of (document, score) tuples
        """
        # Tokenize query and documents
        query_tokens = set(re.findall(r'\w+', query.lower()))
        
        results = []
        for doc in documents:
            doc_tokens = Counter(re.findall(r'\w+', doc.lower()))
            
            # Calculate simple TF score
            score = sum(doc_tokens[token] for token in query_tokens if token in doc_tokens)
            results.append((doc, score))
            
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def vector_search(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """
        Perform vector-based semantic search using embeddings.
        
        Returns: List of (document, similarity_score) tuples
        """
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
    
    def visualize_search_comparison(self, query: str, documents: List[str]):
        """
        Create visualizations comparing keyword and vector search results.
        """
        # Get search results
        keyword_results = self.keyword_search(query, documents)
        vector_results = self.vector_search(query, documents)
        
        # Prepare data for visualization
        keyword_scores = [score for _, score in keyword_results]
        vector_scores = [score for _, score in vector_results]
        
        # Normalize scores for comparison
        max_keyword = max(keyword_scores) if keyword_scores else 1
        keyword_scores = [s/max_keyword for s in keyword_scores]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Keyword search results
        ax1.bar(range(len(documents)), keyword_scores, alpha=0.6)
        ax1.set_title('Keyword Search Results')
        ax1.set_xlabel('Document Index')
        ax1.set_ylabel('Normalized Score')
        
        # Vector search results
        ax2.bar(range(len(documents)), vector_scores, alpha=0.6)
        ax2.set_title('Vector Search Results')
        ax2.set_xlabel('Document Index')
        ax2.set_ylabel('Similarity Score')
        
        plt.tight_layout()
        plt.show()
        
        # Visualize embeddings in 2D
        self.visualize_query_document_space(query, documents)
        
    def visualize_query_document_space(self, query: str, documents: List[str]):
        """
        Create a 2D visualization of query and documents in embedding space.
        """
        # Get embeddings for query and documents
        all_texts = [query] + documents
        embeddings = [self.get_embedding(text) for text in all_texts]
        embeddings_matrix = np.vstack(embeddings)
        
        # Reduce dimensionality for visualization
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings_matrix)
        
        # Create DataFrame for plotting
        df = pd.DataFrame(
            reduced_embeddings,
            columns=['x', 'y']
        )
        df['type'] = ['Query'] + ['Document'] * len(documents)
        df['text'] = all_texts
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='x', y='y', hue='type', style='type', s=100)
        
        # Add text labels
        for idx, row in df.iterrows():
            text = f"Query" if idx == 0 else f"Doc {idx}"
            plt.annotate(text, (row['x'], row['y']), xytext=(5, 5), textcoords='offset points')
            
        plt.title('2D Visualization of Query and Documents in Embedding Space')
        plt.show()

def demonstrate_search_comparison():
    """
    Demonstrate the differences between keyword and semantic search.
    """
    # Initialize comparator (replace with your API key)
    comparator = SearchComparator("YOUR_API_KEY")
    
    # Example corpus with various phrasings and concepts
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
    
    # Example searches to demonstrate differences
    queries = [
        "A fox jumping over a dog",  # Direct phrase match
        "Canines in natural habitats",  # Semantic concept match
        "Sleeping animals outdoors",  # Mixed concept match
        "Forest wildlife activity"  # Thematic match
    ]
    
    # Run demonstrations for each query
    for query in queries:
        print(f"\nAnalyzing search results for query: '{query}'")
        print("\nKeyword Search Results:")
        keyword_results = comparator.keyword_search(query, documents)
        for doc, score in keyword_results[:3]:
            print(f"Score: {score:.4f} | {doc}")
            
        print("\nVector Search Results:")
        vector_results = comparator.vector_search(query, documents)
        for doc, score in vector_results[:3]:
            print(f"Score: {score:.4f} | {doc}")
            
        # Visualize comparisons
        print("\nGenerating visualizations...")
        comparator.visualize_search_comparison(query, documents)

if __name__ == "__main__":
    demonstrate_search_comparison()