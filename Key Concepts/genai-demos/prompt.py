import numpy as np  # For numerical operations and array handling
import matplotlib.pyplot as plt  # For creating visualizations
from sklearn.decomposition import PCA  # For dimensionality reduction (though not used in current code)
import textwrap  # For wrapping text in visualizations
import os  # For file and directory operations
import ollama  # Official Ollama Python client for interacting with Ollama API
from datetime import datetime  # For timestamping output files

def ensure_output_directory():
    """
    Create output directory for visualizations if it doesn't exist.
    
    This function checks if the 'embedding_visualizations' directory exists,
    and creates it if it doesn't. This ensures we have a place to save
    our visualization outputs without raising errors.
    
    Returns:
        str: Path to the output directory
    """
    output_dir = "embedding_visualizations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def save_plot(plt, filename):
    """
    Save the current matplotlib plot to the visualizations directory with timestamp.
    
    This function:
    1. Gets the output directory path
    2. Generates a unique filename with timestamp
    3. Saves the current matplotlib figure
    4. Closes the plot to free up memory
    
    Args:
        plt: The matplotlib pyplot object
        filename (str): Base name for the output file (will be appended with timestamp)
    """
    output_dir = ensure_output_directory()
    # Add timestamp to filename to prevent overwriting previous visualizations
    # Format: YYYYMMDD_HHMMSS (e.g., 20250301_143042)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(output_dir, f"{filename}_{timestamp}.png")
    plt.savefig(full_path)  # Save the figure to the specified path
    print(f"Saved visualization to: {full_path}")
    plt.close()  # Close the plot to free up memory and prevent display overlap

def create_embedding(text):
    """
    Create an embedding for the given text using Ollama's llama3.2:latest model.
    
    This function uses the Ollama Python client to generate an embedding vector
    for the provided text. Embeddings are numerical representations of text that
    capture semantic meaning in a high-dimensional vector space.
    
    The Ollama client handles the API communication details, including:
    - Formatting the request correctly
    - Sending it to the Ollama server
    - Parsing the response
    - Handling potential errors
    
    Args:
        text (str): The text to generate an embedding for
    
    Returns:
        numpy.ndarray: The embedding vector as a numpy array
        
    Notes:
        - The Ollama client must be configured before calling this function
        - The model "llama3.2:latest" must be available in your Ollama installation
        - The returned embedding dimensions depend on the specific model
    """
    # Generate the embedding using the llama3.2:latest model
    # The embeddings() function sends a request to the Ollama API endpoint
    # and returns a dictionary containing the embedding and metadata
    response = ollama.embeddings(
        model="llama3.2:latest",  # Specify which model to use for embedding
        prompt=text  # The text input to embed
    )
    
    # The response contains a key "embedding" with the vector data
    # Convert this to a numpy array for easier mathematical operations
    return np.array(response["embedding"])

def visualize_embedding_stats(embedding):
    """
    Create a visualization of basic statistics about the embedding vector.
    
    This function generates a comprehensive figure with three subplots
    that help analyze different aspects of the embedding vector:
    
    1. Distribution histogram - Shows the spread of values across the vector
    2. Dimension values plot - Shows patterns in the first 50 dimensions
    3. Statistical summary - Shows key numerical properties of the vector
    
    Args:
        embedding (numpy.ndarray): The embedding vector to visualize
    """
    plt.figure(figsize=(12, 4))  # Create a figure with specified width and height
    
    # Plot 1: Histogram of vector values
    plt.subplot(131)  # 1 row, 3 columns, 1st position
    plt.hist(embedding, bins=50)  # Create histogram with 50 bins for detail
    plt.title('Distribution of Vector Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # Plot 2: First 50 dimensions of the vector
    plt.subplot(132)  # 1 row, 3 columns, 2nd position
    plt.plot(embedding[:50])  # Plot only first 50 dimensions for clarity
    plt.title('First 50 Dimensions')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    
    # Plot 3: Basic statistical summary
    # Calculate key statistics about the embedding vector
    stats = f"""
    Mean: {np.mean(embedding):.4f}
    Std: {np.std(embedding):.4f}
    Min: {np.min(embedding):.4f}
    Max: {np.max(embedding):.4f}
    Dimensions: {len(embedding)}
    """
    plt.subplot(133)  # 1 row, 3 columns, 3rd position
    plt.text(0.1, 0.5, stats, fontsize=10)  # Add text at specified position
    plt.axis('off')  # Hide axes for cleaner look
    plt.title('Vector Statistics')
    
    plt.tight_layout()  # Adjust spacing between subplots for better appearance
    save_plot(plt, "embedding_stats")  # Save the visualization

def compare_similar_texts():
    """
    Compare embeddings of semantically similar and different texts.
    
    This function demonstrates how embedding similarity correlates with
    semantic similarity between texts. It:
    
    1. Creates embeddings for a set of test phrases using Ollama
    2. Calculates cosine similarity between all possible pairs
    3. Visualizes the similarity matrix as a heatmap
    
    The test phrases include similar questions about France's capital,
    and a different question about Germany's capital to show contrast.
    This helps visualize how the embedding model captures semantic similarity.
    """
    # Define a set of test phrases to compare
    # First three are semantically related, fourth is different
    texts = [
        "What is the capital of France?",
        "Tell me France's capital city",
        "Paris is located in which country?",
        "What is the capital of Germany?"  # Different meaning
    ]
    
    # Create embeddings for all texts using the Ollama client
    print("Generating embeddings for comparison texts...")
    # List comprehension to get embeddings for each text in the list
    embeddings = [create_embedding(text) for text in texts]
    
    # Define cosine similarity calculation function
    def cosine_similarity(a, b):
        """
        Calculate the cosine similarity between two vectors.
        
        Cosine similarity is defined as the cosine of the angle between two vectors.
        It's a measure of similarity between -1 (exactly opposite) and 1 (exactly the same).
        For embeddings, higher values indicate more similar meanings.
        
        The formula is: cos(θ) = (a·b)/(||a||·||b||)
        
        Args:
            a (numpy.ndarray): First vector
            b (numpy.ndarray): Second vector
            
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        # Numerator: dot product of the vectors
        dot_product = np.dot(a, b)
        # Denominator: product of the L2 norms (vector magnitudes)
        norm_product = np.linalg.norm(a) * np.linalg.norm(b)
        # Return the cosine of the angle between vectors
        return dot_product / norm_product
    
    # Calculate similarity matrix between all pairs of embeddings
    similarities = []
    print("Calculating similarity matrix...")
    for i in range(len(embeddings)):
        row = []
        for j in range(len(embeddings)):
            # Calculate similarity between embedding i and embedding j
            sim = cosine_similarity(embeddings[i], embeddings[j])
            row.append(f"{sim:.3f}")  # Format to 3 decimal places as string
        similarities.append(row)
    
    # Visualize the similarity matrix as a heatmap
    plt.figure(figsize=(10, 8))  # Create figure with adequate size for the heatmap
    
    # Convert string similarities back to float for visualization
    # The imshow function needs numerical values to create the heatmap
    plt.imshow([[float(x) for x in row] for row in similarities], cmap='YlOrRd')
    
    plt.colorbar()  # Add a color scale reference bar
    
    # Add text annotations showing exact similarity values in each cell
    for i in range(len(texts)):
        for j in range(len(texts)):
            plt.text(j, i, similarities[i][j], ha='center', va='center')
    
    # Add wrapped text labels for each axis
    # textwrap.fill breaks long text into multiple lines with specified width
    plt.xticks(range(len(texts)), [textwrap.fill(t, 15) for t in texts], rotation=45)
    plt.yticks(range(len(texts)), [textwrap.fill(t, 15) for t in texts])
    
    plt.title('Cosine Similarity Between Different Prompts')
    plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
    save_plot(plt, "similarity_matrix")  # Save the visualization

def configure_ollama_client():
    """
    Configure the Ollama client to connect to the Ollama server.
    
    This function:
    1. Sets up the Ollama client with the default local server settings
    2. Only asks for a custom URL if running remotely is needed
    
    The Ollama Python client requires a host URL because Ollama runs as
    an HTTP server that the client connects to. By default, it runs on
    localhost port 11434.
    
    Returns:
        None - The function configures the global Ollama client
    """
    # Set default Ollama server location
    default_host = "http://localhost:11434"
    
    # Set up the client with the default host
    ollama.set_host(default_host)
    
    # Ask if user wants to use a non-default Ollama server
    print("\nOllama Connection Configuration")
    print("==============================")
    print(f"Using default Ollama server at {default_host}")
    change_host = input("Connect to a different Ollama server? (y/N): ").lower()
    
    # If user wants to change the host, prompt for new host
    if change_host == 'y' or change_host == 'yes':
        custom_host = input("Enter Ollama server URL: ")
        if custom_host:
            ollama.set_host(custom_host)
            print(f"Now using Ollama server at {custom_host}")
        else:
            print(f"No URL provided, using default {default_host}")

def main():
    """
    Main function to run the embedding visualization workflow.
    
    This function orchestrates the entire process:
    1. Configures the Ollama client
    2. Creates an embedding for a test prompt
    3. Displays basic information about the embedding
    4. Visualizes the embedding statistics
    5. Compares embeddings of similar texts
    
    The workflow demonstrates:
    - How to use the Ollama Python client
    - How to work with embedding vectors
    - How to create informative visualizations
    - How semantic similarity is captured in the embedding space
    """
    print("Embedding Visualization with Ollama and llama3.2:latest")
    print("======================================================")
    print("This script will generate embeddings using Ollama and create")
    print("visualizations to help understand the embedding properties.")
    
    # Configure the Ollama client
    configure_ollama_client()
    
    try:
        # Test prompt for embedding
        text_prompt = "What is the capital of France?"
        print(f"\nCreating embedding for: '{text_prompt}'")
        
        # Create and analyze the embedding
        print("Requesting embedding from Ollama API...")
        embedding = create_embedding(text_prompt)
        
        # Display basic information about the embedding
        print(f"\nEmbedding shape: {embedding.shape}")
        print(f"Number of dimensions: {len(embedding)}")
        print("\nFirst 10 dimensions of the embedding vector:")
        print(embedding[:10])
        
        # Create visualizations
        print("\nVisualizing embedding statistics...")
        visualize_embedding_stats(embedding)
        
        # Compare similar texts
        print("\nComparing similar texts...")
        compare_similar_texts()
        
        print("\nAll visualizations completed successfully!")
        print("Check the 'embedding_visualizations' directory for output files.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting steps:")
        print("=====================")
        print("1. Ensure Ollama is installed and running")
        print("   - Ollama can be installed from https://ollama.com")
        print("   - Check if the Ollama service is running on your system")
        print("\n2. Make sure the llama3.2:latest model is pulled")
        print("   - Run 'ollama pull llama3.2:latest' in your terminal")
        print("   - This may take some time depending on your internet connection")
        print("\n3. Verify the API host is correct")
        print("   - Check for typos in the URL")
        print("   - Ensure the protocol (http://) is included")
        print("   - Confirm the port number is correct (usually 11434)")
        print("\n4. Check that the Ollama Python package is installed")
        print("   - Run 'pip install ollama' in your environment")
        print("   - Ensure you're using the same Python environment as your other packages")

if __name__ == "__main__":
    """
    Entry point of the script.
    
    This conditional ensures the main() function is only executed when 
    the script is run directly (not when imported as a module).
    
    The Python interpreter sets the __name__ variable to "__main__" when
    the file is executed directly, rather than being imported.
    """
    main()