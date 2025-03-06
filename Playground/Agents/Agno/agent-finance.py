# Import necessary components from the Agno library
from agno.agent import Agent  # Core Agent class to create the finance agent
from agno.models.ollama import Ollama  # Ollama integration for using local LLMs
from agno.tools.yfinance import YFinanceTools  # Tools for accessing financial data

# Create a finance agent with specific capabilities
finance_agent = Agent(
    # Identify the agent with a descriptive name
    name="Finance Agent",
    
    # Define the agent's purpose
    description="Your task is to find the finance information",
    
    # Configure the agent to use locally hosted Ollama with llama3.2b model
    # This runs the LLM locally instead of calling cloud APIs
    model=Ollama(id="llama3.2:latest"),
    
    # Provide the agent with specific financial tools from Yahoo Finance
    # Each parameter enables different data retrieval capabilities
    tools=[YFinanceTools(
        stock_price=True,           # Access to current and historical pricing
        analyst_recommendations=True,  # Access to professional stock recommendations
        company_info=True,          # Access to general company information
        company_news=True           # Access to recent news about companies
    )],
    
    # Set specific behavioral instructions for the agent
    instructions=["Use tables to display data"],
    
    # Enable debug features for troubleshooting
    show_tool_calls=True,  # Shows when and how tools are being used
    markdown=True,         # Format responses using markdown for better readability
    debug_mode=True        # Provide detailed execution information
)

# Execute a specific query about NVIDIA analyst recommendations
# The stream=True parameter shows results as they're generated rather than waiting for completion
finance_agent.print_response("Summarize analyst recommendations for NVDA", stream=True)