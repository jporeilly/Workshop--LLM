#!/bin/bash

# Set the application directory
APP_DIR="/app"
API_SERVER="llama-api-server.wasm"
MODEL_FILE="models/llama3-8b-instruct.gguf"
PROMPT_TEMPLATE="llama-3-chat"
SOCKET_ADDR="0.0.0.0:8080"

# Function to check if a file exists
check_file() {
    local file="$1"
    local description="$2"
    
    if [ ! -f "$file" ]; then
        echo "ERROR: $description not found at: $file"
        exit 1
    fi
}

# Function to check if a directory exists
check_directory() {
    local dir="$1"
    local description="$2"
    
    if [ ! -d "$dir" ]; then
        echo "ERROR: $description not found at: $dir"
        exit 1
    fi
}

# Navigate to the app directory
cd "$APP_DIR" || {
    echo "ERROR: Failed to change to directory: $APP_DIR"
    exit 1
}

# Check if we're in the correct directory
current_dir=$(pwd)
if [ "$current_dir" != "$APP_DIR" ]; then
    echo "ERROR: Not in the correct directory. Expected: $APP_DIR, Current: $current_dir"
    exit 1
}

# Check for required files and directories
check_file "$API_SERVER" "LlamaEdge API server WASM file"
check_file "$MODEL_FILE" "LLM model file"

# Check if wasmedge is installed
if ! command -v wasmedge &> /dev/null; then
    echo "ERROR: WasmEdge is not installed or not in PATH"
    exit 1
fi

echo "Starting LlamaEdge API server..."
echo "Using model: $MODEL_FILE"
echo "Using prompt template: $PROMPT_TEMPLATE"
echo "Server will be available on: $SOCKET_ADDR"

# Start the LLama API server with nohup and proper configuration
exec nohup wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:"$MODEL_FILE" \
  "$API_SERVER" \
  --prompt-template "$PROMPT_TEMPLATE" \
  --socket-addr "$SOCKET_ADDR"