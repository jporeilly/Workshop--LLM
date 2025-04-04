"""LangChain utilities and chain definitions."""

from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def create_basic_chain(model: str = "gpt-3.5-turbo", **kwargs: Dict[str, Any]):
    """Create a basic LangChain chain with ChatOpenAI."""
    # Initialize the language model
    llm = ChatOpenAI(model=model, **kwargs)
    
    # Create a basic prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        ("user", "{input}")
    ])
    
    # Create and return the chain
    chain = prompt | llm | StrOutputParser()
    return chain
