
# TechFlow Projects

The project is divded into 3 parts: 
Pt1 - Chatbot:    
Pt2 - RAG:  
Pt3 - Fine Tuning:   

## Pt 1 - Chatbot: Usage Guide

## Dockerfile
Contains configuration for building a Docker container with LLaMA 3 chatbot.

**How to Use:**
1. Build the container:
   ```bash
   docker build -t techflow-chatbot .
   ```
2. Run the container:
   ```bash
   docker run -p 3000:3000 -p 8080:8080 techflow-chatbot
   ```
3. Access the chat interface at `http://localhost:3000`
4. The API server runs on port 8080 if you need direct access

**Note:** Requires Docker installed and about 15GB disk space for the model and container.

## Dockerfile.qwen
Alternative setup using the smaller Qwen model, ideal for systems with less resources.

**How to Use:**
1. Build the Qwen container:
   ```bash
   docker build -f Dockerfile.qwen -t techflow-chatbot .
   ```
2. Run similarly to LLaMA container:
   ```bash
   docker run -p 3000:3000 -p 8080:8080 techflow-chatbot
   ```
3. Access the same way via `http://localhost:3000`

**Key Differences from LLaMA:**
- Smaller model size (3B vs 8B parameters)
- Potentially faster responses
- Lower resource requirements

## cli-inference.md
A command-line interface guide for running direct LLM queries.

**How to Use:**
1. For quick questions, use the basic query command:
   ```bash
   wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm --n-predict 128 --prompt "Your question"
   ```
2. For convenience, set up the provided alias or scripts to avoid typing long commands
3. Use different token lengths based on your needs:
   - 128 tokens: Quick answers
   - 256 tokens: Technical explanations
   - 400 tokens: Code examples and detailed responses

## prompt-template.md
Guide for structuring prompts to get better responses from the models.

**How to Use:**
1. Choose a template based on your needs:
   - Basic technical queries: For quick questions
   - Expert queries: For detailed technical explanations
   - Code review: For analyzing code
   - Troubleshooting: For debugging help

2. Set up the template script:
   ```bash
   chmod +x template-query.sh
   ```

3. Run queries with different roles:
   ```bash
   ./template-query.sh "Your question" expert
   ./template-query.sh "Your question" review
   ./template-query.sh "Your question" debug
   ```

**Tips:**
- Use the expert role for detailed technical explanations
- Use the review role when analyzing code or configurations
- Use the debug role for step-by-step troubleshooting
- Default role (general) is good for basic questions

**Note:** These templates work with both LLaMA and Qwen models, but you'll need to modify the model path in the commands based on which one you're using.