import os
import sys
import logging
import subprocess
import requests
import time
from typing import Iterator, List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get Qdrant URL from environment variables
qdrant_url = os.getenv("QDRANT_URL_LOCALHOST", "http://localhost:6333")

# Define model for embeddings - using a smaller, efficient sentence transformer model
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Set the export type - DOC_CHUNKS means documents will be exported as chunked pieces
EXPORT_TYPE = ExportType.DOC_CHUNKS

# Path to the PDF file that will be processed
FILE_PATH = "./data/DeepSeek_R1.pdf"  

# Target Ollama model
TARGET_MODEL = "deepseek-llm:latest"

def check_ollama_running() -> bool:
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama server: {str(e)}")
        return False

def check_qdrant_running() -> bool:
    """Check if Qdrant server is running"""
    try:
        response = requests.get(f"{qdrant_url}/collections", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except Exception as e:
        logger.error(f"Error checking Qdrant server: {str(e)}")
        return False

def start_qdrant_container():
    """Attempt to start a Qdrant container if it's not running"""
    try:
        # Check if Docker is available
        docker_check = subprocess.run(["docker", "--version"], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE,
                                     text=True)
        
        if docker_check.returncode != 0:
            logger.error("Docker is not available. Please install Docker or start Qdrant manually.")
            return False
        
        # Check if qdrant container exists
        container_check = subprocess.run(["docker", "ps", "-a", "--filter", "name=qdrant", "--format", "{{.Names}}"],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       text=True)
        
        container_exists = "qdrant" in container_check.stdout
        
        if container_exists:
            # Start existing container
            logger.info("Found existing Qdrant container. Attempting to start it...")
            subprocess.run(["docker", "start", "qdrant"])
        else:
            # Create and start new container
            logger.info("Creating new Qdrant container...")
            subprocess.run([
                "docker", "run", "-d",
                "--name", "qdrant",
                "-p", "6333:6333",
                "-p", "6334:6334",
                "-v", "qdrant_storage:/qdrant/storage",
                "qdrant/qdrant"
            ])
        
        # Wait for container to start
        for _ in range(5):
            if check_qdrant_running():
                logger.info("Qdrant server is now running")
                return True
            logger.info("Waiting for Qdrant server to start...")
            time.sleep(2)
        
        logger.error("Qdrant server failed to start within the expected time")
        return False
        
    except Exception as e:
        logger.error(f"Error starting Qdrant container: {str(e)}")
        return False

def list_available_models() -> List[str]:
    """List all available models in Ollama"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model.get("name") for model in models]
        return []
    except requests.exceptions.ConnectionError:
        return []
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return []

def pull_model(model_name: str) -> bool:
    """Pull a model from Ollama"""
    logger.info(f"Pulling model: {model_name}")
    try:
        # Using subprocess to show progress in real-time
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
        
        process.wait()
        return process.returncode == 0
    except Exception as e:
        logger.error(f"Error pulling model: {str(e)}")
        return False

def setup_ollama_model() -> bool:
    """Ensure the required Ollama model is available"""
    # Check if Ollama is running
    if not check_ollama_running():
        logger.error("Ollama server is not running. Please start it first.")
        return False
    
    # List available models
    models = list_available_models()
    logger.info(f"Available models: {', '.join(models) if models else 'None'}")
    
    # Check for DeepSeek model
    if any(model.startswith("deepseek") for model in models):
        logger.info(f"DeepSeek model already available")
        return True
    else:
        logger.info(f"DeepSeek model not found. Will pull {TARGET_MODEL}")
        success = pull_model(TARGET_MODEL)
        if success:
            logger.info(f"Successfully pulled {TARGET_MODEL}")
            return True
        else:
            logger.error(f"Failed to pull {TARGET_MODEL}")
            return False

def update_application_model() -> Optional[str]:
    """
    Check for the application file and update the model reference if needed.
    Returns the file path if updated successfully, None otherwise.
    """
    app_file = "./rag-chainlit-deepseek.py"
    
    if not os.path.exists(app_file):
        logger.warning(f"Application file {app_file} not found. You will need to manually update your model reference.")
        return None
    
    try:
        with open(app_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for the Ollama initialization pattern
        if "deepseek-r1:latest" in content:
            # Replace the incorrect model name with the correct one
            updated_content = content.replace("deepseek-r1:latest", TARGET_MODEL)
            
            # Write the updated content back
            with open(app_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            logger.info(f"Updated model reference in {app_file} from 'deepseek-r1:latest' to '{TARGET_MODEL}'")
            return app_file
        else:
            logger.info(f"No incorrect model reference found in {app_file} or the file uses a different format.")
            return None
    except Exception as e:
        logger.error(f"Error updating application file: {str(e)}")
        return None

def create_vector_database():
    """
    Creates a vector database from a PDF document using Docling for document processing
    and Qdrant for vector storage.
    
    The function:
    1. Loads and chunks the document using Docling
    2. Processes chunks based on export type
    3. Saves the content to a markdown file
    4. Creates embeddings using HuggingFace
    5. Stores the embeddings in a Qdrant vector database
    """
    
    # Check if Qdrant is running
    if not check_qdrant_running():
        logger.error("Qdrant server is not running. Attempting to start it...")
        if not start_qdrant_container():
            logger.error("""
            Failed to connect to Qdrant server. 
            
            Please ensure Qdrant is installed and running:
            
            To install and run Qdrant with Docker:
              docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
            
            Or run Qdrant locally following instructions at:
              https://qdrant.tech/documentation/quick-start/
            """)
            return False
    
    # Initialize DoclingLoader with specified parameters
    loader = DoclingLoader(
        file_path=FILE_PATH,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(
            tokenizer=EMBED_MODEL_ID,
            chunk_size=300,  # Reduced chunk size
            chunk_overlap=30,  # Some overlap to maintain context between chunks
            split_factor=0.5,  # More aggressive splitting
        ),
    )
    
    logger.info(f"Loading document from {FILE_PATH}")
    
    # Load and process the document
    docling_documents = loader.load()
    
    # Process the documents based on the export type
    if EXPORT_TYPE == ExportType.DOC_CHUNKS:
        # Create a text splitter with a smaller chunk size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # Characters, not tokens, but a safe size
            chunk_overlap=50,
            length_function=lambda text: len(text.split()),  # Approximate token count using word count
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Further split any chunks that might be too large
        logger.info(f"Processing {len(docling_documents)} initial chunks from Docling")
        splits = []
        for doc in docling_documents:
            # Ensure metadata has a 'page' field to avoid KeyError('page')
            if 'page' not in doc.metadata:
                # Extract page number from source if available, or default to 1
                page_num = doc.metadata.get('source', '').split('_')[-1].split('.')[0] if 'source' in doc.metadata else '1'
                try:
                    doc.metadata['page'] = int(page_num)
                except ValueError:
                    doc.metadata['page'] = 1
            
            # Check if this chunk is potentially too large
            if len(doc.page_content.split()) > 400:  # If chunk has > 400 words, further split it
                logger.info(f"Splitting large chunk of size ~{len(doc.page_content.split())} words")
                smaller_chunks = text_splitter.split_text(doc.page_content)
                # Convert the text chunks back to LangChain Documents with metadata preserved
                splits.extend([
                    LCDocument(page_content=chunk, metadata=doc.metadata) 
                    for chunk in smaller_chunks
                ])
            else:
                splits.append(doc)
        
        logger.info(f"After additional splitting: {len(splits)} chunks")
        
    elif EXPORT_TYPE == ExportType.MARKDOWN:
        # Split based on markdown headers
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        initial_splits = [split for doc in docling_documents for split in splitter.split_text(doc.page_content)]
        
        # Further chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=lambda text: len(text.split()),
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = []
        for doc in initial_splits:
            # Ensure metadata has a 'page' field
            if 'page' not in doc.metadata:
                doc.metadata['page'] = 1  # Default page value
                
            if len(doc.page_content.split()) > 400:
                smaller_chunks = text_splitter.split_text(doc.page_content)
                splits.extend([
                    LCDocument(page_content=chunk, metadata=doc.metadata) 
                    for chunk in smaller_chunks
                ])
            else:
                splits.append(doc)
                
        logger.info(f"After markdown splitting and additional chunking: {len(splits)} chunks")
    else:
        # Raise an error for unsupported export types
        raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")
    
    
    # Save the processed document to a markdown file
    with open('data/output_docling.md', 'a', encoding='utf-8') as f:  # utf-8 encoding for unicode support
        for doc in docling_documents:
            f.write(doc.page_content + '\n')
    
    
    # Initialize the embedding model from HuggingFace
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    
    # Log metadata of documents for debugging
    for i, doc in enumerate(splits[:3]):  # Log first 3 documents as samples
        logger.info(f"Document {i} metadata: {doc.metadata}")
    
    # Check for extremely long chunks before embedding
    max_token_length = max([len(doc.page_content.split()) for doc in splits])
    logger.info(f"Longest chunk is approximately {max_token_length} words")
    
    if max_token_length > 500:
        logger.warning(f"Some chunks may still be too long for the embedding model (max: {max_token_length} words)")
    
    # Create a Qdrant vector store from the document chunks
    try:
        # Final check to ensure Qdrant is still running
        if not check_qdrant_running():
            logger.error("Qdrant server connection lost. Please ensure the server is running properly.")
            return False
            
        # Create the vector store
        logger.info(f"Creating vector store at {qdrant_url}")
        vectorstore = QdrantVectorStore.from_documents(
            documents=splits,
            embedding=embedding,
            url=qdrant_url,
            collection_name="rag",
            force_recreate=True,  # Force recreate the collection if it exists
        )
        logger.info(f"Successfully created vector store with {len(splits)} chunks")
        return True
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        logger.error(f"""
        Failed to connect to Qdrant at {qdrant_url}.
        
        Troubleshooting steps:
        1. Check if Qdrant is running: 
           - For Docker: run 'docker ps' to see if the container is running
           - For local installation: check if the process is running
        
        2. Verify the URL in your .env file:
           - It should contain QDRANT_URL_LOCALHOST=http://localhost:6333
        
        3. Make sure ports 6333 and 6334 are not blocked by a firewall
        
        4. Try running Qdrant manually:
           - Docker: docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
           - Local: follow instructions at https://qdrant.tech/documentation/quick-start/
        """)
        return False

def save_documents_to_pickle(documents, output_file="data/processed_documents.pkl"):
    """Save processed documents to a pickle file as a fallback"""
    import pickle
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(documents, f)
        logger.info(f"Saved processed documents to {output_file} as a fallback")
        return True
    except Exception as e:
        logger.error(f"Error saving documents to pickle: {str(e)}")
        return False

def main():
    """Main function to run the complete RAG setup"""
    logger.info("Starting RAG setup process...")
    
    # Step 1: Set up Ollama model
    logger.info("STEP 1: Setting up Ollama model...")
    if not setup_ollama_model():
        logger.error("Failed to set up Ollama model. Exiting.")
        return
    
    # Step 2: Update application model reference if needed
    logger.info("STEP 2: Checking application model reference...")
    updated_file = update_application_model()
    if updated_file:
        logger.info(f"Successfully updated model reference in {updated_file}")
    else:
        logger.warning("No automatic model update performed. Please check your application file manually.")
    
    # Step 3: Create vector database
    logger.info("STEP 3: Creating vector database...")
    db_success = create_vector_database()
    
    if db_success:
        logger.info("Vector database created successfully!")
        # Final step: Display success message
        logger.info("""
        =====================================================
        RAG SETUP COMPLETED SUCCESSFULLY!
        
        What's been done:
        1. Checked/pulled the DeepSeek LLM model in Ollama
        2. Updated application file model reference (if found)
        3. Created vector database with proper metadata
        
        You can now run your Chainlit application:
        $ chainlit run rag-chainlit-deepseek.py
        =====================================================
        """)
    else:
        logger.error("""
        =====================================================
        PARTIAL SETUP COMPLETED
        
        What's been done:
        1. Checked/pulled the DeepSeek LLM model in Ollama ✓
        2. Updated application file model reference ✓
        3. Failed to create vector database ✗
        
        Please troubleshoot your Qdrant database connection
        before running your Chainlit application.
        =====================================================
        """)

if __name__ == "__main__":
    main()