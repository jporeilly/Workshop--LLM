import os
# import nest_asyncio  # noqa: E402
# nest_asyncio.apply()
from typing import Iterator
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get Qdrant URL from environment variables
qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")

# Define model for embeddings - using a smaller, efficient sentence transformer model
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Set the export type - DOC_CHUNKS means documents will be exported as chunked pieces
EXPORT_TYPE = ExportType.DOC_CHUNKS

# Path to the PDF file that will be processed
FILE_PATH = "./data/DeepSeek_R1.pdf"
            
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
    
    # Initialize DoclingLoader with specified parameters:
    # - file_path: Path to the PDF document
    # - export_type: How to export the document (as chunks)
    # - chunker: HybridChunker using the specified embedding model for tokenization
    loader = DoclingLoader(
        file_path=FILE_PATH,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    )
    
    # Load and process the document
    docling_documents = loader.load()
    
    # Process the documents based on the export type
    if EXPORT_TYPE == ExportType.DOC_CHUNKS:
        # If using DOC_CHUNKS, use the chunks as they are
        splits = docling_documents
    elif EXPORT_TYPE == ExportType.MARKDOWN:
        # If using MARKDOWN, split the document based on markdown headers
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        # Apply the splitter to each document and flatten the result
        splits = [split for doc in docling_documents for split in splitter.split_text(doc.page_content)]
    else:
        # Raise an error for unsupported export types
        raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")
    
    
    # Save the processed document to a markdown file for reference/debugging
    with open('data/output_docling.md', 'a') as f:  # 'a' mode appends to file (doesn't overwrite)
        for doc in docling_documents:
            f.write(doc.page_content + '\n')
    
    
    # Initialize the embedding model from HuggingFace
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    
    # Create a Qdrant vector store from the document chunks:
    # - documents: The processed document chunks
    # - embedding: The embedding model to use
    # - url: The URL of the Qdrant server
    # - collection_name: The name of the collection in Qdrant
    vectorstore = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embedding,
        url=qdrant_url,
        collection_name="rag",  # RAG = Retrieval Augmented Generation
    )
    
    print('Vector DB created successfully!')

# Check if this script is being run directly (not imported)
if __name__ == "__main__":
    create_vector_database()