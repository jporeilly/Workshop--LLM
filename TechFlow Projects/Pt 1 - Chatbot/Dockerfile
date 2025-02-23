# Base image setup
FROM ubuntu:22.04

# Prevent interactive prompts during build
# These environment variables ensure that apt doesn't wait for user input
ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y

# Install system dependencies and development tools
# This section installs all needed packages in a single RUN command to reduce image layers
# - software-properties-common: Adds apt repository management
# - build-essential: Provides compilation tools (gcc, make)
# - libopenblas-dev: Optimized linear algebra library for AI performance
# - ninja-build: Fast build system alternative to make
# - Additional tools: git, curl, node.js, etc.
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
    && rm -rf /var/lib/apt/lists/*  # Cleanup to reduce image size

# Set up Git LFS for handling large files
# Required for efficiently working with large ML model files
RUN git lfs install

# Install and configure WasmEdge for WebAssembly runtime
# WasmEdge is a lightweight container runtime for WebAssembly
# This installs the runtime and adds it to the system PATH
RUN curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash
ENV PATH="/root/.wasmedge/bin:${PATH}"
ENV WASMEDGE_HOME="/root/.wasmedge"

# Create directory structure for application components
# Setting up a clean structure for the various application parts
WORKDIR /app
RUN mkdir -p models scripts logs chatbot-ui

# Download the Qwen model
# Qwen is a large language model (LLM) from Alibaba
# Using the GGUF format which is optimized for inference
WORKDIR /app/models
RUN curl -L "https://huggingface.co/second-state/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q5_K_S.gguf" -o qwen-3b-instruct.gguf

# Set up LlamaEdge components
# LlamaEdge provides WebAssembly-based inference for LLMs
# Downloading the inference engine and API server components
WORKDIR /app
RUN curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-simple.wasm && \
    curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm

# Install chat UI interface
# Downloads and extracts the frontend UI into the proper directory
# http-server will be used to serve the static UI files
RUN curl -L https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz | \
    tar xz -C /app/chatbot-ui --strip-components=1
RUN npm install -g http-server

# Verify key component installations
# This helps confirm the build is successful and debug installation issues
RUN wasmedge --version && \
    npm --version && \
    node --version

# Reset package manager to normal mode
# No longer need to suppress interactive prompts after installation is complete
ENV DEBIAN_FRONTEND=dialog

# Set working directory for runtime
WORKDIR /app

# Create startup script that:
# - Initializes logging
# - Starts API server with Qwen model
# - Verifies API server is running
# - Launches UI server with proxy to API
# This complex heredoc creates a bash script with proper error handling
RUN echo '#!/bin/bash\n\
\n\
# Function to log messages with timestamps\n\
log_message() {\n\
    echo "[$(date '"'"'+%Y-%m-%d %H:%M:%S'"'"')] $1"\n\
}\n\
\n\
# Create log files\n\
mkdir -p /app/logs\n\
touch /app/logs/api.log\n\
touch /app/logs/ui.log\n\
\n\
# Start API server in background\n\
log_message "Starting API server..."\n\
cd /app && wasmedge --dir .:. \\\n\
    --nn-preload default:GGML:AUTO:models/qwen-3b-instruct.gguf \\\n\
    llama-api-server.wasm \\\n\
    --prompt-template chatml \\\n\
    --socket-addr "0.0.0.0:8080" > /app/logs/api.log 2>&1 &\n\
API_PID=$!\n\
\n\
# Wait for API to be available\n\
log_message "Waiting for API server to start..."\n\
max_attempts=30\n\
count=0\n\
while [ $count -lt $max_attempts ]; do\n\
    if curl -s "http://localhost:8080/v1/models" > /dev/null 2>&1; then\n\
        log_message "API server is up and running"\n\
        break\n\
    fi\n\
    count=$((count + 1))\n\
    log_message "Attempt $count/$max_attempts: API server not ready yet..."\n\
    sleep 2\n\
done\n\
\n\
if [ $count -eq $max_attempts ]; then\n\
    log_message "ERROR: API server failed to start"\n\
    cat /app/logs/api.log\n\
    exit 1\n\
fi\n\
\n\
# Start UI server\n\
log_message "Starting UI server..."\n\
cd /app/chatbot-ui && http-server . \\\n\
    --port 3000 \\\n\
    --cors \\\n\
    --gzip \\\n\
    --proxy "http://localhost:8080" \\\n\
    --proxy-options.secure=false > /app/logs/ui.log 2>&1\n\
' > /app/scripts/entrypoint.sh && chmod +x /app/scripts/entrypoint.sh

# Configure persistent storage and networking
# VOLUME instruction creates a mount point for persistent data
# EXPOSE documents the ports the container will listen on
VOLUME ["/app/logs"]
EXPOSE 8080 3000

# Set container startup command
# This runs the entrypoint script when the container starts
ENTRYPOINT ["/app/scripts/entrypoint.sh"]