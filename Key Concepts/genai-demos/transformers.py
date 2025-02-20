from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd
import os
from datetime import datetime
from getpass import getpass

def ensure_output_directory() -> str:
    """Create and return the output directory path with timestamp."""
    base_dir = "transformer_analysis"
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

class TransformerDemonstrator:
    """
    Demonstrates transformer processing using OpenAI embeddings
    """
    def __init__(self, api_key: str, output_dir: str):
        """Initialize the demonstrator with API key and output directory."""
        self.client = OpenAI(api_key=api_key)
        self.output_dir = output_dir
        self.prompt = "What is the capital of France?"
        self.tokens = ['What', 'is', 'the', 'capital', 'of', 'France', '?']
        self.response = "Paris"
        
        # Create results file
        self.results_file = os.path.join(output_dir, "analysis_results.txt")
        
    def save_results(self, section: str, content: str):
        """Save analysis results to the results file."""
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{section}\n")
            f.write("=" * len(section) + "\n")
            f.write(content + "\n")
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings from OpenAI API"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)
    
    def save_visualization(self, fig, filename: str) -> str:
        """Save visualization to the output directory."""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath)
        plt.close(fig)
        print(f"Saved visualization to: {filepath}")
        return filepath
    
    def demonstrate_process(self):
        """Demonstrate the complete transformer process"""
        # Save initial configuration
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
                token_embeddings[token] = self.get_embeddings(token)
            
            embeddings_info = "Generated embeddings for tokens:\n"
            for token in self.tokens:
                embedding = token_embeddings[token]
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
            error_msg = f"Error during demonstration: {str(e)}"
            print(f"\nError: {error_msg}")
            self.save_results("Error Log", error_msg)
            raise
        
    def visualize_token_attention(self, token_embeddings: Dict[str, np.ndarray]):
        """Visualize attention between tokens"""
        n_tokens = len(self.tokens)
        attention_matrix = np.zeros((n_tokens, n_tokens))
        
        # Simulate attention scores based on token relationships
        for i, token1 in enumerate(self.tokens):
            for j, token2 in enumerate(self.tokens):
                # Get embeddings for token pair
                emb1 = token_embeddings[token1]
                emb2 = token_embeddings[token2]
                # Calculate cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                attention_matrix[i, j] = similarity
        
        # Normalize attention scores
        attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
        
        # Save attention matrix data
        attention_info = "Attention Matrix:\n"
        for i, token1 in enumerate(self.tokens):
            for j, token2 in enumerate(self.tokens):
                attention_info += f"{token1} -> {token2}: {attention_matrix[i,j]:.4f}\n"
        self.save_results("Token Attention", attention_info)
        
        # Create visualization
        fig = plt.figure(figsize=(12, 8))
        sns.heatmap(attention_matrix, 
                   annot=True, 
                   fmt='.2f', 
                   xticklabels=self.tokens,
                   yticklabels=self.tokens,
                   cmap='YlOrRd')
        plt.title('Token Self-Attention Weights')
        plt.xlabel('Context Tokens')
        plt.ylabel('Query Tokens')
        plt.tight_layout()
        
        self.save_visualization(fig, 'token_attention.png')
        
    def visualize_transformer_stages(self):
        """Visualize the stages of transformer processing"""
        stages = [
            'Input Embedding',
            'Positional Encoding',
            'Self-Attention',
            'Feed Forward',
            'Layer Normalization',
            'Final Representation'
        ]
        
        # Save stages information
        stages_info = "Transformer Processing Stages:\n"
        for i, stage in enumerate(stages):
            stages_info += f"{i+1}. {stage}\n"
        self.save_results("Processing Stages", stages_info)
        
        # Create visualization showing information flow
        fig = plt.figure(figsize=(15, 8))
        for i, stage in enumerate(stages):
            plt.barh(i, 0.8, color='skyblue', alpha=0.6)
            plt.text(0.9, i, stage, va='center')
            
            # Add arrows between stages
            if i < len(stages) - 1:
                plt.arrow(0.4, i, 0, 0.8, head_width=0.05, 
                         head_length=0.1, fc='k', ec='k')
        
        plt.ylim(-0.5, len(stages) - 0.5)
        plt.xlim(0, 2)
        plt.title('Transformer Processing Stages')
        plt.axis('off')
        plt.tight_layout()
        
        self.save_visualization(fig, 'transformer_stages.png')
        
    def visualize_response_process(self):
        """Visualize the response generation process"""
        # Get embeddings for prompt and response
        prompt_emb = self.get_embeddings(self.prompt)
        response_emb = self.get_embeddings(self.response)
        
        # Save embeddings information
        response_info = f"""
        Prompt: '{self.prompt}'
        - Embedding shape: {prompt_emb.shape}
        - Embedding mean: {np.mean(prompt_emb):.4f}
        
        Response: '{self.response}'
        - Embedding shape: {response_emb.shape}
        - Embedding mean: {np.mean(response_emb):.4f}
        """
        self.save_results("Response Generation", response_info)
        
        # Create visualization showing relationship
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prompt processing
        ax1.bar(['Prompt'], [1], color='lightblue')
        ax1.set_title('Input Processing')
        ax1.text(0, 0.5, self.prompt, ha='center', va='center')
        
        # Response generation
        ax2.bar(['Response'], [1], color='lightgreen')
        ax2.set_title('Output Generation')
        ax2.text(0, 0.5, self.response, ha='center', va='center')
        
        plt.tight_layout()
        
        self.save_visualization(fig, 'response_generation.png')

def demonstrate_full_process():
    """Run complete transformer demonstration"""
    try:
        # Create output directory
        output_dir = ensure_output_directory()
        print(f"\nAnalysis results will be saved to: {output_dir}")
        
        # Get API key securely
        api_key = get_api_key()
        
        # Initialize demonstrator
        demonstrator = TransformerDemonstrator(api_key, output_dir)
        
        print("\nDemonstrating Transformer Process:")
        print(f"Input Prompt: '{demonstrator.prompt}'")
        
        # Run demonstration
        demonstrator.demonstrate_process()
        
        print(f"\nAll analysis results have been saved to: {output_dir}")
        print("Generated files:")
        print("1. token_attention.png - Shows attention weights between tokens")
        print("2. transformer_stages.png - Shows stages of transformer processing")
        print("3. response_generation.png - Shows response generation process")
        print("4. analysis_results.txt - Detailed analysis data and metrics")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    demonstrate_full_process()