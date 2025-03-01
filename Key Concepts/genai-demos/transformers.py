import numpy as np  # For numerical operations and array handling
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns  # For enhanced visualizations (especially heatmaps)
from typing import List, Dict  # Type hints for better code documentation
import pandas as pd  # For data manipulation (used in some visualizations)
import os  # For file and directory operations
from datetime import datetime  # For timestamping output files
import ollama  # Python client for interacting with Ollama API

def ensure_output_directory() -> str:
    """
    Create and return the output directory path with timestamp.
    
    This function creates a unique timestamped directory for each run to prevent
    overwriting previous results and provide easy identification.
    
    Returns:
        str: Path to the created output directory
    """
    # Base directory for transformer analysis outputs
    base_dir = "transformer_analysis"
    
    # Generate timestamp for unique directory name (format: YYYYMMDD_HHMMSS)
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

class TransformerDemonstrator:
    """
    Demonstrates transformer processing using Ollama embeddings.
    
    This class provides methods to visualize and understand how transformers work,
    using the llama3.2:latest model from Ollama to generate embeddings and simulate
    the transformer process.
    """
    def __init__(self, ollama_host: str, output_dir: str):
        """
        Initialize the demonstrator with Ollama host and output directory.
        
        Args:
            ollama_host: URL of the Ollama API server
            output_dir: Directory to save visualizations and analysis results
        """
        # Initialize the Ollama client with the specified host
        self.client = ollama.Client(host=ollama_host)
        
        # Specify which Ollama model to use for embeddings
        self.model = "llama3.2:latest"
        
        # Directory where all output files will be saved
        self.output_dir = output_dir
        
        # Example prompt, tokens, and response for demonstration
        self.prompt = "What is the capital of France?"
        self.tokens = ['What', 'is', 'the', 'capital', 'of', 'France', '?']
        self.response = "Paris"
        
        # Create results file path
        self.results_file = os.path.join(output_dir, "analysis_results.txt")
        
        # Initialize the results file with header
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write(f"Transformer Analysis Results\n")
            f.write("=========================\n")
            f.write(f"Model: Ollama - {self.model}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
    def save_results(self, section: str, content: str):
        """
        Save analysis results to the results file.
        
        This function appends a new section of results to the analysis file
        with proper formatting and section headers.
        
        Args:
            section: Title of the section being added
            content: Text content to save in that section
        """
        # Open file in append mode
        with open(self.results_file, 'a', encoding='utf-8') as f:
            # Add section header with underline
            f.write(f"\n{section}\n")
            f.write("=" * len(section) + "\n")
            # Write the actual content
            f.write(content + "\n")
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get embeddings from Ollama API.
        
        This function sends a request to Ollama to generate an embedding vector
        for the provided text using the llama3.2:latest model.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        # Request embedding from Ollama API
        response = self.client.embeddings(
            model=self.model,  # Using llama3.2:latest model
            prompt=text  # The text to embed
        )
        
        # Convert the embedding to numpy array
        return np.array(response["embedding"])
    
    def save_visualization(self, fig, filename: str) -> str:
        """
        Save visualization to the output directory.
        
        Args:
            fig: Matplotlib figure to save
            filename: Name for the saved file
            
        Returns:
            str: Path to the saved file
        """
        # Create full path for the output file
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the figure
        fig.savefig(filepath)
        
        # Close the figure to free memory
        plt.close(fig)
        
        print(f"Saved visualization to: {filepath}")
        return filepath
    
    def demonstrate_process(self):
        """
        Demonstrate the complete transformer process.
        
        This method orchestrates the visualization of different aspects of
        transformer architecture using our example prompt:
        1. Token embeddings
        2. Self-attention between tokens
        3. Transformer processing stages
        4. Response generation
        """
        # Save initial configuration information
        config_info = f"""
        Input Prompt: '{self.prompt}'
        Tokens: {self.tokens}
        Expected Response: '{self.response}'
        """
        self.save_results("Configuration", config_info)
        
        try:
            # 1. Get embeddings for each token
            token_embeddings = {}
            print("\nGenerating embeddings for tokens...")
            for token in self.tokens:
                # Get embedding for each token and store in dictionary
                token_embeddings[token] = self.get_embeddings(token)
            
            # Save embedding information to results file
            embeddings_info = "Generated embeddings for tokens:\n"
            for token in self.tokens:
                embedding = token_embeddings[token]
                # Record shape and basic statistics for each embedding
                embeddings_info += f"{token}: Shape {embedding.shape}, Mean {np.mean(embedding):.4f}\n"
            self.save_results("Token Embeddings", embeddings_info)
            
            # 2. Visualize token attention
            print("\nGenerating token attention visualization...")
            self.visualize_token_attention(token_embeddings)
            
            # 3. Visualize transformer stages
            print("\nGenerating transformer stages visualization...")
            self.visualize_transformer_stages()
            
            # 4. Visualize response generation
            print("\nGenerating response process visualization...")
            self.visualize_response_process()
            
        except Exception as e:
            # Log any errors that occur
            error_msg = f"Error during demonstration: {str(e)}"
            print(f"\nError: {error_msg}")
            self.save_results("Error Log", error_msg)
            raise
        
    def visualize_token_attention(self, token_embeddings: Dict[str, np.ndarray]):
        """
        Visualize attention between tokens.
        
        This method simulates the self-attention mechanism in transformers by calculating
        similarity scores between token embeddings and visualizing them as a heatmap.
        
        Args:
            token_embeddings: Dictionary mapping tokens to their embedding vectors
        """
        # Get the number of tokens
        n_tokens = len(self.tokens)
        
        # Create empty matrix to store attention scores
        attention_matrix = np.zeros((n_tokens, n_tokens))
        
        # Calculate attention scores based on token embeddings similarity
        # In transformers, attention is based on query-key compatibility
        # We simulate this using cosine similarity between token embeddings
        for i, token1 in enumerate(self.tokens):
            for j, token2 in enumerate(self.tokens):
                # Get embeddings for token pair
                emb1 = token_embeddings[token1]  # Query token
                emb2 = token_embeddings[token2]  # Key token
                
                # Calculate cosine similarity
                # Formula: cos(θ) = (a·b)/(||a||·||b||)
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                # Store in attention matrix
                attention_matrix[i, j] = similarity
        
        # Normalize attention scores to sum to 1 for each query token (row)
        # This simulates the softmax operation in transformer attention
        attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
        
        # Save attention matrix data to results file
        attention_info = "Attention Matrix:\n"
        for i, token1 in enumerate(self.tokens):
            for j, token2 in enumerate(self.tokens):
                attention_info += f"{token1} -> {token2}: {attention_matrix[i,j]:.4f}\n"
        self.save_results("Token Attention", attention_info)
        
        # Create visualization using seaborn's heatmap
        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention_matrix, 
            annot=True,           # Show values in each cell
            fmt='.2f',            # Format as 2 decimal places
            xticklabels=self.tokens,  # Labels for columns (key tokens)
            yticklabels=self.tokens,  # Labels for rows (query tokens)
            cmap='YlOrRd'         # Color map: yellow to orange to red
        )
        plt.title('Token Self-Attention Weights')
        plt.xlabel('Context Tokens (Keys)')
        plt.ylabel('Query Tokens')
        plt.tight_layout()
        
        # Save the visualization
        self.save_visualization(fig, 'token_attention.png')
        
    def visualize_transformer_stages(self):
        """
        Visualize the stages of transformer processing.
        
        This method creates a diagram showing the main processing stages
        in a transformer model.
        """
        # Define the main stages of transformer processing
        stages = [
            'Input Embedding',        # Convert tokens to vectors
            'Positional Encoding',    # Add position information
            'Self-Attention',         # Compute attention between tokens
            'Feed Forward',           # Process through neural network
            'Layer Normalization',    # Normalize activations
            'Final Representation'    # Output token representations
        ]
        
        # Save stages information to results file
        stages_info = "Transformer Processing Stages:\n"
        for i, stage in enumerate(stages):
            stages_info += f"{i+1}. {stage}\n"
        self.save_results("Processing Stages", stages_info)
        
        # Create visualization showing information flow between stages
        fig = plt.figure(figsize=(15, 8))
        
        # For each stage, create a horizontal bar and label
        for i, stage in enumerate(stages):
            plt.barh(i, 0.8, color='skyblue', alpha=0.6)
            plt.text(0.9, i, stage, va='center')
            
            # Add arrows between stages to show information flow
            if i < len(stages) - 1:
                plt.arrow(0.4, i, 0, 0.8, head_width=0.05, 
                         head_length=0.1, fc='k', ec='k')
        
        # Set plot limits and title
        plt.ylim(-0.5, len(stages) - 0.5)
        plt.xlim(0, 2)
        plt.title('Transformer Processing Stages')
        plt.axis('off')  # Hide axes
        plt.tight_layout()
        
        # Save the visualization
        self.save_visualization(fig, 'transformer_stages.png')
        
    def visualize_response_process(self):
        """
        Visualize the response generation process.
        
        This method shows the relationship between the input prompt
        and the generated response using embeddings to represent them.
        """
        # Get embeddings for the full prompt and the response
        print("Generating embeddings for prompt and response...")
        prompt_emb = self.get_embeddings(self.prompt)
        response_emb = self.get_embeddings(self.response)
        
        # Save embeddings information to results file
        response_info = f"""
        Prompt: '{self.prompt}'
        - Embedding shape: {prompt_emb.shape}
        - Embedding mean: {np.mean(prompt_emb):.4f}
        - Embedding std: {np.std(prompt_emb):.4f}
        
        Response: '{self.response}'
        - Embedding shape: {response_emb.shape}
        - Embedding mean: {np.mean(response_emb):.4f}
        - Embedding std: {np.std(response_emb):.4f}
        
        Cosine Similarity between prompt and response: 
        {np.dot(prompt_emb, response_emb) / (np.linalg.norm(prompt_emb) * np.linalg.norm(response_emb)):.4f}
        """
        self.save_results("Response Generation", response_info)
        
        # Create visualization showing relationship between prompt and response
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prompt processing visualization (left subplot)
        ax1.bar(['Prompt'], [1], color='lightblue')
        ax1.set_title('Input Processing')
        ax1.text(0, 0.5, self.prompt, ha='center', va='center')
        
        # Response generation visualization (right subplot)
        ax2.bar(['Response'], [1], color='lightgreen')
        ax2.set_title('Output Generation')
        ax2.text(0, 0.5, self.response, ha='center', va='center')
        
        # Add title and adjust layout
        plt.suptitle('Transformer Input/Output Process', fontsize=16)
        plt.tight_layout()
        
        # Save the visualization
        self.save_visualization(fig, 'response_generation.png')

def demonstrate_full_process():
    """
    Run complete transformer demonstration.
    
    This function sets up the environment, initializes the demonstrator,
    and runs the full transformer process demonstration.
    """
    print("Transformer Visualization Demo using Ollama")
    print("===========================================")
    print("This script demonstrates transformer processing using")
    print("the llama3.2:latest model via Ollama.\n")
    
    try:
        # Create output directory
        output_dir = ensure_output_directory()
        print(f"\nAnalysis results will be saved to: {output_dir}")
        
        # Get Ollama host configuration
        ollama_host = get_ollama_host()
        
        # Initialize demonstrator
        print(f"\nInitializing TransformerDemonstrator with Ollama (model: llama3.2:latest)")
        demonstrator = TransformerDemonstrator(ollama_host, output_dir)
        
        print("\nDemonstrating Transformer Process:")
        print(f"Input Prompt: '{demonstrator.prompt}'")
        
        # Run demonstration
        demonstrator.demonstrate_process()
        
        print(f"\nAll analysis results have been saved to: {output_dir}")
        print("\nGenerated files:")
        print("1. token_attention.png - Shows attention weights between tokens")
        print("2. transformer_stages.png - Shows stages of transformer processing")
        print("3. response_generation.png - Shows response generation process")
        print("4. analysis_results.txt - Detailed analysis data and metrics")
        
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
    demonstrate_full_process()