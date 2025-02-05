# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Configure environment to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y

# Update system and install required dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    software-properties-common \
    build-essential \
    libopenblas-dev \
    ninja-build \
    pkg-config \
    cmake-data \
    clang \
    nano \
    git \
    git-lfs \
    curl \
    wget \
    zip \
    unzip \
    nodejs \
    npm \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git Large File Storage
RUN git lfs install

# Install WasmEdge runtime
RUN curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash

# Configure WasmEdge environment variables
ENV PATH="/root/.wasmedge/bin:${PATH}"
ENV WASMEDGE_HOME="/root/.wasmedge"

# Create base directory structure
WORKDIR /app
RUN mkdir -p models scripts logs chatbot-ui

# Download LLM model
WORKDIR /app/models
RUN curl -L "https://huggingface.co/second-state/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf" -o llama3-8b-instruct.gguf

# Download WASM binaries for LlamaEdge
WORKDIR /app
RUN curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-simple.wasm && \
    curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm

# Download, extract ChatUI interface
RUN curl -L https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz | \
    tar xz -C /app/chatbot-ui --strip-components=1

# Install HTTP server for UI
RUN npm install -g http-server

# Verify installations
RUN wasmedge --version && \
    npm --version && \
    node --version

# Reset package manager interface
ENV DEBIAN_FRONTEND=dialog

# Set default working directory
WORKDIR /app

# Create startup script
RUN echo '#!/bin/bash\n\
# Copy scripts from mounted volume\n\
cp /home/pentaho/TechFlow-Chatbot/scripts/start-api.sh /app/scripts/\n\
cp /home/pentaho/TechFlow-Chatbot/scripts/start-ui.sh /app/scripts/\n\
\n\
# Convert line endings\n\
dos2unix /app/scripts/start-api.sh\n\
dos2unix /app/scripts/start-ui.sh\n\
\n\
# Make scripts executable\n\
chmod +x /app/scripts/start-api.sh\n\
chmod +x /app/scripts/start-ui.sh\n\
\n\
# Create log files\n\
touch /app/logs/api.log\n\
touch /app/logs/ui.log\n\
\n\
# Start API server with output to log\n\
cd /app && /app/scripts/start-api.sh > /app/logs/api.log 2>&1\n\
\n\
# Start UI server\n\
cd /app/chatbot-ui && /app/scripts/start-ui.sh\n\
' > /app/scripts/entrypoint.sh && chmod +x /app/scripts/entrypoint.sh

# Mount points for scripts, data, and logs
VOLUME ["/home/pentaho/TechFlow-Chatbot"]
VOLUME ["/app/logs"]

# Expose ports for API and UI servers
EXPOSE 8080
EXPOSE 3000

# Use the entrypoint script
ENTRYPOINT ["/app/scripts/entrypoint.sh"]/logs