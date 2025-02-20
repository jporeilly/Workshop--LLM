from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd
import os

class TransformerDemonstrator:
    """
    Demonstrates transformer processing using OpenAI embeddings
    """
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.prompt = "What is the capital of France?"
        self.tokens = ['What', 'is', 'the', 'capital', 'of', 'France', '?']
        self.response = "Paris"
        
        # Create visualization directory
        self.viz_dir = "transformer_visualizations"
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
            print(f"Created visualization directory: {self.viz_dir}")
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings from OpenAI API"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)
    
    def save_visualization(self, filename: str):
        """Save visualization to the dedicated folder"""
        filepath = os.path.join(self.viz_dir, filename)
        plt.savefig(filepath)
        plt.close()
        return filepath
    
    def demonstrate_process(self):
        """Demonstrate the complete transformer process"""
        # 1. Get embeddings for each token
        token_embeddings = {}
        print("\nGenerating embeddings for tokens...")
        for token in self.tokens:
            token_embeddings[token] = self.get_embeddings(token)
        
        # 2. Visualize token attention
        filepath = self.visualize_token_attention()
        print(f"Saved token attention visualization: {filepath}")
        
        # 3. Visualize transformer stages
        filepath = self.visualize_transformer_stages()
        print(f"Saved transformer stages visualization: {filepath}")
        
        # 4. Visualize response generation
        filepath = self.visualize_response_process()
        print(f"Saved response generation visualization: {filepath}")
        
    def visualize_token_attention(self):
        """Visualize attention between tokens"""
        n_tokens = len(self.tokens)
        attention_matrix = np.zeros((n_tokens, n_tokens))
        
        # Simulate attention scores based on token relationships
        for i, token1 in enumerate(self.tokens):
            for j, token2 in enumerate(self.tokens):
                # Get embeddings for token pair
                emb1 = self.get_embeddings(token1)
                emb2 = self.get_embeddings(token2)
                # Calculate cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                attention_matrix[i, j] = similarity
        
        # Normalize attention scores
        attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
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
        
        return self.save_visualization('token_attention.png')
        
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
        
        # Create visualization showing information flow
        plt.figure(figsize=(15, 8))
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
        
        return self.save_visualization('transformer_stages.png')
        
    def visualize_response_process(self):
        """Visualize the response generation process"""
        # Get embeddings for prompt and response
        prompt_emb = self.get_embeddings(self.prompt)
        response_emb = self.get_embeddings(self.response)
        
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
        
        return self.save_visualization('response_generation.png')

def demonstrate_full_process():
    """
    Run complete transformer demonstration
    """
    # Initialize demonstrator (replace with your API key)
    demonstrator = TransformerDemonstrator("YOUR_API_KEY")
    
    print("Demonstrating Transformer Process:")
    print(f"\nInput Prompt: '{demonstrator.prompt}'")
    
    # Run demonstration
    demonstrator.demonstrate_process()
    
    print("\nAll visualizations have been saved in the 'transformer_visualizations' folder:")
    print("1. token_attention.png - Shows attention weights between tokens")
    print("2. transformer_stages.png - Shows stages of transformer processing")
    print("3. response_generation.png - Shows response generation process")
    
if __name__ == "__main__":
    demonstrate_full_process()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd

class TransformerDemonstrator:
    """
    Demonstrates transformer processing using OpenAI embeddings
    """
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.prompt = "What is the capital of France?"
        self.tokens = ['What', 'is', 'the', 'capital', 'of', 'France', '?']
        self.response = "Paris"
        
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings from OpenAI API"""
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        return np.array(response.data[0].embedding)
    
    def demonstrate_process(self):
        """Demonstrate the complete transformer process"""
        # 1. Get embeddings for each token
        token_embeddings = {}
        print("\nGenerating embeddings for tokens...")
        for token in self.tokens:
            token_embeddings[token] = self.get_embeddings(token)
        
        # 2. Visualize token attention
        self.visualize_token_attention()
        
        # 3. Visualize transformer stages
        self.visualize_transformer_stages()
        
        # 4. Visualize response generation
        self.visualize_response_process()
        
    def visualize_token_attention(self):
        """Visualize attention between tokens"""
        n_tokens = len(self.tokens)
        attention_matrix = np.zeros((n_tokens, n_tokens))
        
        # Simulate attention scores based on token relationships
        for i, token1 in enumerate(self.tokens):
            for j, token2 in enumerate(self.tokens):
                # Get embeddings for token pair
                emb1 = self.get_embeddings(token1)
                emb2 = self.get_embeddings(token2)
                # Calculate cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                attention_matrix[i, j] = similarity
        
        # Normalize attention scores
        attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
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
        plt.savefig('token_attention.png')
        plt.close()
        
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
        
        # Create visualization showing information flow
        plt.figure(figsize=(15, 8))
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
        plt.savefig('transformer_stages.png')
        plt.close()
        
    def visualize_response_process(self):
        """Visualize the response generation process"""
        # Get embeddings for prompt and response
        prompt_emb = self.get_embeddings(self.prompt)
        response_emb = self.get_embeddings(self.response)
        
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
        plt.savefig('response_generation.png')
        plt.close()

def demonstrate_full_process():
    """
    Run complete transformer demonstration
    """
    # Initialize demonstrator (replace with your API key)
    demonstrator = TransformerDemonstrator("YOUR_API_KEY")
    
    print("Demonstrating Transformer Process:")
    print(f"\nInput Prompt: '{demonstrator.prompt}'")
    
    # Run demonstration
    demonstrator.demonstrate_process()
    
    print("\nVisualizations saved:")
    print("- token_attention.png: Shows attention weights between tokens")
    print("- transformer_stages.png: Shows stages of transformer processing")
    print("- response_generation.png: Shows response generation process")
    
if __name__ == "__main__":
    demonstrate_full_process()