"""Streamlit application for GenAI."""

import os

import streamlit as st
from dotenv import load_dotenv

from genai.chains import create_basic_chain

# Load environment variables
load_dotenv()

def main():
    """Main Streamlit application."""
    st.title("GenAI Demo")
    
    # Input field for user query
    user_input = st.text_input("Enter your query:")
    
    if user_input:
        # Create and run the chain
        chain = create_basic_chain()
        with st.spinner("Thinking..."):
            response = chain.invoke({"input": user_input})
            st.write(response)

if __name__ == "__main__":
    main()
