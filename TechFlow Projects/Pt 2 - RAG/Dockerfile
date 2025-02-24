# Start with Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to prevent interactive prompts during build
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

# Initialize Git LFS
RUN git lfs install

# Install WasmEdge with specific version
RUN curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash
ENV PATH="/root/.wasmedge/bin:${PATH}"
ENV WASMEDGE_HOME="/root/.wasmedge"

# Create directory structure
WORKDIR /app
RUN mkdir -p models scripts logs chatbot-ui qdrant/storage qdrant/snapshots

# Download ML models
WORKDIR /app/models
RUN curl -L "https://huggingface.com/second-state/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q5_K_S.gguf" -o qwen-3b-instruct.gguf && \
    curl -L "https://huggingface.com/gaianet/Nomic-embed-text-v1.5-Embedding-GGUF/resolve/main/nomic-embed-text-v1.5.f16.gguf" -o nomic-embed-text-v1.5.f16.gguf

# Download LlamaEdge components with specific version
WORKDIR /app
RUN curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-simple.wasm && \
    curl -LO https://github.com/LlamaEdge/rag-api-server/releases/latest/download/rag-api-server.wasm && \
    chmod +x llama-simple.wasm && \
    chmod +x rag-api-server.wasm && \
    # Verify downloads are valid WASM files
    file llama-simple.wasm && \
    file rag-api-server.wasm

# Install chat UI interface
# Downloads and extracts the frontend UI into the proper directory
# http-server will be used to serve the static UI files
RUN curl -L https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz | \
    tar xz -C /app/chatbot-ui --strip-components=1
RUN npm install -g http-server

# Install Qdrant with fixed version
RUN cd /tmp && \
    curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz -o qdrant.tar.gz && \
    tar xvf qdrant.tar.gz && \
    mv qdrant /usr/local/bin/ && \
    rm qdrant.tar.gz

# Verify installations
RUN wasmedge --version && \
    npm --version && \
    node --version && \
    qdrant --version && \
    ls -la /app/*.wasm

# Reset package manager to normal mode
ENV DEBIAN_FRONTEND=dialog

# Set working directory
WORKDIR /app

# Create the entrypoint script with fixed Qdrant configuration
RUN echo '#!/bin/bash\n\
\n\
log_message() {\n\
    echo "[$(date '"'"'+%Y-%m-%d %H:%M:%S'"'"')] $1"\n\
}\n\
\n\
mkdir -p /app/logs /app/qdrant/storage /app/qdrant/snapshots\n\
touch /app/logs/api.log\n\
touch /app/logs/ui.log\n\
touch /app/logs/qdrant.log\n\
\n\
# Create Qdrant config\n\
cat > /app/qdrant/config.yaml << EOF\n\
storage:\n\
  storage_path: /app/qdrant/storage\n\
  snapshots_path: /app/qdrant/snapshots\n\
  on_disk_payload: true\n\
service:\n\
  host: 0.0.0.0\n\
  http_port: 6333\n\
  grpc_port: 6334\n\
EOF\n\
\n\
# Start Qdrant server\n\
log_message "Starting Qdrant server..."\n\
qdrant --config-path /app/qdrant/config.yaml > /app/logs/qdrant.log 2>&1 &\n\
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
# Start API server\n\
log_message "Starting API server..."\n\
cd /app && wasmedge --dir .:. rag-api-server.wasm \\\n\
    --model-name qwen-3b-instruct.gguf \\\n\
    --model-dir /app/models \\\n\
    --embedding-name nomic-embed-text-v1.5.f16.gguf \\\n\
    --qdrant-url http://localhost:6333 \\\n\
    --collection-name pentaho \\\n\
    --qdrant-create-collection \\\n\
    --qdrant-vector-dimension 768 \\\n\
    --rag \\\n\
    --ctx-size 16384 > /app/logs/api.log 2>&1 &\n\
API_PID=$!\n\
\n\
# Wait for API\n\
log_message "Waiting for API server..."\n\
max_attempts=30\n\
count=0\n\
while [ $count -lt $max_attempts ]; do\n\
    if curl -s "http://localhost:8080/v1/models" > /dev/null 2>&1; then\n\
        log_message "API server is running"\n\
        break\n\
    fi\n\
    count=$((count + 1))\n\
    log_message "Attempt $count/$max_attempts: API not ready..."\n\
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

# Define volumes for persistent storage
VOLUME ["/app/logs", "/app/qdrant/storage", "/app/qdrant/snapshots"]

# Expose ports
EXPOSE 8080 3000 6333 6334

# Set entrypoint
ENTRYPOINT ["/app/scripts/entrypoint.sh"]