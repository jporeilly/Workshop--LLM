# Import standard library modules
import os  # For environment variable access and file operations

# Import type hints for better code documentation
from typing import Iterable  # For type hinting collections that can be iterated over

# Import LangChain document handling
from langchain_core.documents import Document as LCDocument  # Core document class for LangChain

# Import LangChain prompt handling
from langchain.prompts import ChatPromptTemplate  # For creating structured prompts for chat models

# Import embedding model from HuggingFace integration
from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # For text-to-vector conversions

# Import LangChain runnable components for building the pipeline
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig  # For creating processing pipelines
from langchain.schema import StrOutputParser  # For parsing LLM outputs as strings

# Import callback handling for tracking pipeline operations
from langchain.callbacks.base import BaseCallbackHandler  # Base class for creating custom callbacks

# Import Ollama integration for accessing local LLMs
from langchain_ollama import OllamaLLM  # For interfacing with locally running Ollama models

# Import Qdrant vector database integration
from langchain_qdrant import QdrantVectorStore  # For connecting to Qdrant vector database

# Import Chainlit for building the chat interface
import chainlit as cl  # Web-based chat interface framework


# Load environment variables from .env file
from dotenv import load_dotenv  # For loading environment variables from a .env file
load_dotenv()  # Execute the loading of environment variables


# Get the Qdrant database URL from environment variables
qdrant_url = os.getenv("QDRANT_URL_LOCALHOST")  # URL for the local Qdrant instance

# Define the embedding model to use - this converts text to vector embeddings
# all-MiniLM-L6-v2 is a lightweight, efficient embedding model with good performance
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


# Initialize the language model using Ollama
# deepseek-r1 is the specific model being used for generating responses
llm = OllamaLLM(
    model="deepseek-r1:latest"  # Using the latest version of the deepseek-r1 model
)


# This decorator registers this function to run when a new chat session starts
@cl.on_chat_start
async def on_chat_start():
    # Define the prompt template that instructs the LLM how to answer
    # {context} will be filled with retrieved documents
    # {question} will be filled with the user's query
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    # Create a prompt object from the template string
    prompt = ChatPromptTemplate.from_template(template)

    # Helper function to format a list of documents into a single string
    # This combines all retrieved document contents with newlines as separators
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # Initialize the embedding model for converting text to vectors
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    
    # Connect to an existing Qdrant collection named "rag"
    # This collection should already contain embedded documents
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embedding,  # The embedding model to use for query conversion
        collection_name="rag",  # The name of the collection in Qdrant
        url=qdrant_url  # The URL of the Qdrant server
    )
    
    # Create a retriever from the vector store
    # This will be used to find relevant documents based on query similarity
    retriever = vectorstore.as_retriever()

    # Build the RAG pipeline using LangChain's runnable interface
    # This defines the sequence of operations that will process each user query
    runnable = (
        # Step 1: Prepare inputs for the prompt template
        {
            # Retrieve documents and format them into a single context string
            "context": retriever | format_docs, 
            # Pass the user's question through unchanged
            "question": RunnablePassthrough()
        }
        # Step 2: Fill the prompt template with the context and question
        | prompt
        # Step 3: Send the filled prompt to the language model
        | llm
        # Step 4: Parse the LLM output as a string
        | StrOutputParser()
    )

    # Store the runnable in the user's session for reuse with each message
    cl.user_session.set("runnable", runnable)
    
    
# This decorator registers this function to handle each new user message
@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the runnable from the user's session
    runnable = cl.user_session.get("runnable")  # type: Runnable
    
    # Create an empty message that will be populated with the response
    # The content will be streamed in chunks as it's generated
    msg = cl.Message(content="")

    # Define a custom callback handler to track and display document sources
    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg  # Store reference to the message being built
            self.sources = set()  # Use a set to store unique source-page pairs

        # This method is called when document retrieval is complete
        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            # Extract source and page information from each retrieved document
            for d in documents:
                source_page_pair = (d.metadata['source'], d.metadata['page'])
                self.sources.add(source_page_pair)  # Add unique pairs to the set

        # This method is called when the LLM finishes generating a response
        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            # If we have sources to display, format them and add as an element
            if len(self.sources):
                # Create a formatted string of sources with page references
                sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
                # Add the sources as a text element to the message
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    # Stream the response from the runnable, processing the user's message
    async for chunk in runnable.astream(
        message.content,  # Pass the user's message content to the runnable
        config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(),  # Standard Chainlit-LangChain integration
            PostMessageHandler(msg)  # Our custom handler for tracking sources
        ]),
    ):
        # Stream each token to the UI as it's generated
        await msg.stream_token(chunk)

    # Send the complete message once streaming is finished
    await msg.send()
