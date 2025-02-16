# Base image setup
FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV ACCEPT_EULA=Y

# Install system dependencies and development tools
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

# Set up Git LFS for handling large files
RUN git lfs install

# Install and configure WasmEdge
RUN curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash
ENV PATH="/root/.wasmedge/bin:${PATH}"
ENV WASMEDGE_HOME="/root/.wasmedge"

# Create directory structure
WORKDIR /app
RUN mkdir -p models scripts logs chatbot-ui qdrant/storage qdrant/snapshots

# Download models
WORKDIR /app/models
RUN curl -L "https://huggingface.co/second-state/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf" -o llama3-8b-instruct.gguf && \
    curl -L "https://huggingface.co/gaianet/Nomic-embed-text-v1.5-Embedding-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf" -o nomic-embed-text-v1.5.f16.gguf

# Set up LlamaEdge components and RAG API server
WORKDIR /app
RUN curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-simple.wasm && \
    curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/rag-api-server.wasm

# Install chat UI interface
RUN curl -L https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz | \
    tar xz -C /app/chatbot-ui --strip-components=1
RUN npm install -g http-server

# Install Qdrant
RUN curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-linux-x86_64.tar.gz | \
    tar xz -C /usr/local/bin

# Verify installations
RUN wasmedge --version && \
    npm --version && \
    node --version && \
    qdrant --version

ENV DEBIAN_FRONTEND=dialog

WORKDIR /app

# Updated startup script with RAG API server
RUN echo '#!/bin/bash\n\
\n\
log_message() {\n\
    echo "[$(date '"'"'+%Y-%m-%d %H:%M:%S'"'"')] $1"\n\
}\n\
\n\
mkdir -p /app/logs\n\
touch /app/logs/api.log\n\
touch /app/logs/ui.log\n\
touch /app/logs/qdrant.log\n\
\n\
# Start Qdrant server\n\
log_message "Starting Qdrant server..."\n\
qdrant --storage-path /app/qdrant/storage --snapshots-path /app/qdrant/snapshots > /app/logs/qdrant.log 2>&1 &\n\
QDRANT_PID=$!\n\
\n\
# Wait for Qdrant\n\
log_message "Waiting for Qdrant server..."\n\
max_attempts=30\n\
count=0\n\
while [ $count -lt $max_attempts ]; do\n\
    if curl -s "http://localhost:6333/health" > /dev/null 2>&1; then\n\
        log_message "Qdrant server is running"\n\
        break\n\
    fi\n\
    count=$((count + 1))\n\
    log_message "Attempt $count/$max_attempts: Qdrant not ready..."\n\
    sleep 2\n\
done\n\
\n\
if [ $count -eq $max_attempts ]; then\n\
    log_message "ERROR: Qdrant server failed to start"\n\
    cat /app/logs/qdrant.log\n\
    exit 1\n\
fi\n\
\n\
# Start RAG API server\n\
log_message "Starting RAG API server..."\n\
cd /app && wasmedge --dir .:. \\\n\
    --nn-preload default:GGML:AUTO:models/llama3-8b-instruct.gguf \\\n\
    --nn-preload embedding:GGML:AUTO:models/nomic-embed-text-v1.5.f16.gguf \\\n\
    rag-api-server.wasm \\\n\
    --prompt-template llama-3-chat,embedding \\\n\
    --model-name llama31,nomic-embed \\\n\
    --ctx-size 16384,8192 \\\n\
    --batch-size 128,8192 \\\n\
    --rag-policy system-message \\\n\
    --rag-prompt "Use the following pieces of context to answer the user'"'"'s question.\nIf you don'"'"'t know the answer, just say that you don'"'"'t know, don'"'"'t try to make up an answer.\n----------------\n" \\\n\
    --qdrant-url http://127.0.0.1:6333 \\\n\
    --qdrant-collection-name "pentaho" \\\n\
    --qdrant-limit 1 \\\n\
    --qdrant-score-threshold 0.2 \\\n\
    --socket-addr 0.0.0.0:8080 \\\n\
    --log-prompts > /app/logs/api.log 2>&1 &\n\
API_PID=$!\n\
\n\
# Wait for RAG API\n\
log_message "Waiting for RAG API server..."\n\
max_attempts=30\n\
count=0\n\
while [ $count -lt $max_attempts ]; do\n\
    if curl -s "http://localhost:8080/v1/models" > /dev/null 2>&1; then\n\
        log_message "RAG API server is running"\n\
        break\n\
    fi\n\
    count=$((count + 1))\n\
    log_message "Attempt $count/$max_attempts: RAG API not ready..."\n\
    sleep 2\n\
done\n\
\n\
if [ $count -eq $max_attempts ]; then\n\
    log_message "ERROR: RAG API server failed to start"\n\
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

# Configure volumes and ports
VOLUME ["/app/logs", "/app/qdrant/storage", "/app/qdrant/snapshots"]
EXPOSE 8080 3000 6333 6334

ENTRYPOINT ["/app/scripts/entrypoint.sh"]