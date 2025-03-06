# Import the dedent function to format multi-line strings without extra indentation
from textwrap import dedent

# Import necessary components from the agno library
# Agent: Core class for creating AI agents
# RunResponse: Used for handling responses from the agent (marked with noqa to ignore linting)
from agno.agent import Agent, RunResponse  # noqa
# Import the Ollama model class which provides access to local LLM models
from agno.models.ollama import Ollama

# Create a new Agent instance with a news reporter personality
# This instantiates our AI agent with specific model and behavior instructions
agent = Agent(
    # Specify which model to use - here we're using deepseek-llm from Ollama
    # Ollama is a tool for running local LLMs, and deepseek-llm is a specific model available through it
    model=Ollama(id="deepseek-llm:latest"), 
    
    # Define the agent's personality and behavior using a multi-line string
    # dedent() removes the leading indentation from the multi-line string to improve readability
    instructions=dedent("""\
        You are an enthusiastic news reporter with a flair for storytelling! 🗽
        Think of yourself as a mix between a witty comedian and a sharp journalist.

        Your style guide:
        - Start with an attention-grabbing headline using emoji
        - Share news with enthusiasm and NYC attitude
        - Keep your responses concise but entertaining
        - Throw in local references and NYC slang when appropriate
        - End with a catchy sign-off like 'Back to you in the studio!' or 'Reporting live from the Big Apple!'

        Remember to verify all facts while keeping that NYC energy high!\
    """),
    
    # Enable markdown formatting in the agent's responses
    # This allows for rich text formatting like bold, italics, and headings
    markdown=True,
)

# Execute the agent with a test prompt
# This will print the agent's response to the console in real-time (streaming)
agent.print_response(
    "Tell me about a breaking news story happening in Times Square.", 
    stream=True  # Enable streaming mode to see responses character-by-character
)

# A commented-out section containing additional example prompts
# These are suggestions for further testing the agent with different scenarios
"""
Try these fun scenarios:
1. "What's the latest food trend taking over Brooklyn?"
2. "Tell me about a peculiar incident on the subway today"
3. "What's the scoop on the newest rooftop garden in Manhattan?"
4. "Report on an unusual traffic jam caused by escaped zoo animals"
5. "Cover a flash mob wedding proposal at Grand Central"
"""