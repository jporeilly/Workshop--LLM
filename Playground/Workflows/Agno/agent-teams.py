from agno.agent import Agent
from agno.models.ollama import OllamaChat  # Import OllamaChat for local LLM integration
from agno.playground import Playground, serve_playground_app
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Define storage location for agent conversations
agent_storage: str = "tmp/agents.db"

# Configure Ollama model integration
# --------------------------------
# This creates an instance of OllamaChat that connects to a locally running 
# Ollama server. The Ollama server must be running separately and have 
# the specified model (llama3.2b:latest) already pulled/available.
# No separate Ollama Python client import is needed as Agno's OllamaChat
# handles the API communication directly.
llama_model = OllamaChat(
    id="llama3.2b:latest",  # Specifies which model to use from Ollama
    # No need to specify base_url as it defaults to http://localhost:11434
)

# Web Agent configuration
# ----------------------
# This agent can search the web using DuckDuckGo
web_agent = Agent(
    name="Web Agent",
    model=llama_model,  # Use local Llama model instead of OpenAI
    tools=[DuckDuckGoTools()],  # Provide web search capabilities
    instructions=["Always include sources"],  # Custom instruction for the agent
    # Persistence configuration
    storage=SqliteAgentStorage(table_name="web_agent", db_file=agent_storage),
    # Additional agent settings
    add_datetime_to_instructions=True,  # Adds current date/time context
    add_history_to_messages=True,  # Includes conversation history
    num_history_responses=5,  # How many past exchanges to include
    markdown=True,  # Format responses as markdown
)

# Finance Agent configuration
# --------------------------
# This agent can retrieve financial data using YFinance
finance_agent = Agent(
    name="Finance Agent",
    model=llama_model,  # Use local Llama model instead of OpenAI
    # Provide access to various financial data tools
    tools=[YFinanceTools(
        stock_price=True, 
        analyst_recommendations=True, 
        company_info=True, 
        company_news=True
    )],
    instructions=["Always use tables to display data"],  # Custom instruction
    # Use same storage pattern but different table
    storage=SqliteAgentStorage(table_name="finance_agent", db_file=agent_storage),
    # Same additional settings as web agent
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

# Create and configure the playground application
app = Playground(agents=[web_agent, finance_agent]).get_app()

# Run the application when script is executed directly
if __name__ == "__main__":
    # Start the playground with hot-reloading enabled for development
    serve_playground_app("playground:app", reload=True)