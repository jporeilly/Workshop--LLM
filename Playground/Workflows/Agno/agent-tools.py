# Import the Agent class from the agno.agent module - this is the main class for creating and managing AI agents
from agno.agent import Agent

# Import the Ollama class from agno.models.ollama - this provides integration with Ollama, 
# a framework for running large language models locally
from agno.models.ollama import Ollama

# Import DuckDuckGoTools which provides search functionality through the DuckDuckGo search engine
from agno.tools.duckduckgo import DuckDuckGoTools

# Create a new Agent instance with the following configuration:
agent = Agent(
    # Set the language model to use - in this case, Ollama with the llama3.2:latest model
    # This specifies we want to use Llama 3.2 (Meta's LLM) served through Ollama
    model=Ollama(id="llama3.2:latest"),
    
    # Provide a list of tools the agent can use - here, just DuckDuckGoTools for web search capabilities
    # This allows the agent to search the internet for current information
    tools=[DuckDuckGoTools()],
    
    # When set to True, this displays the tool calls in the output so you can see when and how
    # the agent is using tools like search
    show_tool_calls=True,
    
    # Enable markdown formatting in the agent's responses for better readability
    markdown=True,
)

# Use the agent to generate a response to the query "What's happening in France?"
# The stream=True parameter means the response will be displayed incrementally as it's generated,
# rather than waiting for the complete response before showing anything
agent.print_response("Whats happening in France?", stream=True)
