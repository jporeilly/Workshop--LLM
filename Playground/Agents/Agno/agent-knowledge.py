from textwrap import dedent  # For clean multi-line string formatting

# Import required components from the agno framework
from agno.agent import Agent  # Core Agent class that orchestrates the entire process
from agno.embedder.ollama import OllamaEmbedder  # Embeds text using Ollama models
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase  # Loads knowledge from PDF URLs
from agno.models.ollama import Ollama  # Interface to Ollama's local LLM models
from agno.tools.duckduckgo import DuckDuckGoTools  # Enables web searches via DuckDuckGo
from agno.vectordb.lancedb import LanceDb, SearchType  # Vector database for storing embeddings

# Create a specialized Thai Recipe Expert Agent
agent = Agent(
    # Configure the language model - using local Llama 3.2 via Ollama
    model=Ollama(id="llama3.2:latest"),  # Local LLM without API calls to OpenAI
    
    # Detailed instructions that define the agent's personality and behavior
    instructions=dedent("""\
        You are a passionate and knowledgeable Thai cuisine expert! 🧑‍🍳
        Think of yourself as a combination of a warm, encouraging cooking instructor,
        a Thai food historian, and a cultural ambassador.

        Follow these steps when answering questions:
        1. First, search the knowledge base for authentic Thai recipes and cooking information
        2. If the information in the knowledge base is incomplete OR if the user asks a question better suited for the web, search the web to fill in gaps
        3. If you find the information in the knowledge base, no need to search the web
        4. Always prioritize knowledge base information over web results for authenticity
        5. If needed, supplement with web searches for:
            - Modern adaptations or ingredient substitutions
            - Cultural context and historical background
            - Additional cooking tips and troubleshooting

        Communication style:
        1. Start each response with a relevant cooking emoji
        2. Structure your responses clearly:
            - Brief introduction or context
            - Main content (recipe, explanation, or history)
            - Pro tips or cultural insights
            - Encouraging conclusion
        3. For recipes, include:
            - List of ingredients with possible substitutions
            - Clear, numbered cooking steps
            - Tips for success and common pitfalls
        4. Use friendly, encouraging language

        Special features:
        - Explain unfamiliar Thai ingredients and suggest alternatives
        - Share relevant cultural context and traditions
        - Provide tips for adapting recipes to different dietary needs
        - Include serving suggestions and accompaniments

        End each response with an uplifting sign-off like:
        - 'Happy cooking! ขอให้อร่อย (Enjoy your meal)!'
        - 'May your Thai cooking adventure bring joy!'
        - 'Enjoy your homemade Thai feast!'

        Remember:
        - Always verify recipe authenticity with the knowledge base
        - Clearly indicate when information comes from web sources
        - Be encouraging and supportive of home cooks at all skill levels\
    """),
    
    # Configure the knowledge base with Thai recipe information
    knowledge=PDFUrlKnowledgeBase(
        # Source PDF containing Thai recipes stored in S3
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        
        # Vector database configuration for efficient semantic search
        vector_db=LanceDb(
            uri="tmp/lancedb",  # Local storage location for the vector database
            table_name="recipe_knowledge",  # Name of the table within LanceDB
            search_type=SearchType.hybrid,  # Uses both keyword and semantic search for better results
            # Configure the embedder to convert text to vectors using Ollama
            embedder=OllamaEmbedder(
                id="llama3.2",  # Using Llama 3.2 model for creating embeddings
                dimensions=3072,  # Specifies the embedding vector size for Llama 3.2
            ),
        ),
    ),
    
    # Add web search capability using DuckDuckGo
    tools=[DuckDuckGoTools()],  # Allows the agent to search the web for supplementary information
    
    # Additional configuration options
    show_tool_calls=True,  # Shows when external tools like web search are being used
    markdown=True,  # Formats responses using markdown for better readability
    add_references=True,  # Includes references to sources of information in responses
)

# Ensure the knowledge base is loaded before making queries
# This step may be time-consuming on first run as it downloads and processes the PDF
if agent.knowledge is not None:
    agent.knowledge.load()  # Loads the PDF, extracts text, creates embeddings, and stores in the vector DB

# Example queries with streaming responses (prints tokens as they're generated)
# Query 1: Request for a specific Thai soup recipe
agent.print_response(
    "How do I make chicken and galangal in coconut milk soup", stream=True
)

# Query 2: Request for historical information about Thai curry
agent.print_response("What is the history of Thai curry?", stream=True)

# Query 3: Request for ingredients list for a popular Thai dish
agent.print_response("What ingredients do I need for Pad Thai?", stream=True)
