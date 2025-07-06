#!/bin/bash

# GPU Configuration Helper for AI Stack
# This script helps you optimize GPU allocation across services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[DETAIL]${NC} $1"
}

echo "🔧 GPU Configuration Helper for AI Stack"
echo "========================================"

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

# Get GPU information
print_status "Detecting GPU configuration..."
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
print_status "Found $GPU_COUNT GPU(s)"

echo ""
echo "🖥️  GPU Details:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r index name memory_total memory_free utilization; do
    echo "  GPU $index: $name"
    echo "    Memory: ${memory_free}MB free / ${memory_total}MB total"
    echo "    Utilization: ${utilization}%"
    echo ""
done

# Backup existing .env
if [ -f .env ]; then
    cp .env .env.backup
    print_status "Backed up existing .env to .env.backup"
fi

echo "🎯 GPU Allocation Strategy:"
echo "=========================="
echo "1. Single GPU (use all services on GPU 0)"
echo "2. Dual GPU (Ollama + Open-WebUI on GPU 0, Flowise + others on GPU 1)"
echo "3. Multi-GPU (distribute services across available GPUs)"
echo "4. Custom (manually specify GPU assignments)"
echo "5. Show current configuration"

read -p "Choose your strategy (1-5): " -n 1 -r
echo ""

case $REPLY in
    1)
        print_status "Configuring for single GPU setup..."
        sed -i '/^NVIDIA_VISIBLE_DEVICES=/c\NVIDIA_VISIBLE_DEVICES=0' .env
        sed -i '/^# FLOWISE_GPU_DEVICE=/c\FLOWISE_GPU_DEVICE=0' .env
        sed -i '/^# OPEN_WEBUI_GPU_DEVICE=/c\OPEN_WEBUI_GPU_DEVICE=0' .env
        sed -i '/^# N8N_GPU_DEVICE=/c\N8N_GPU_DEVICE=0' .env
        sed -i '/^# QDRANT_GPU_DEVICE=/c\QDRANT_GPU_DEVICE=0' .env
        print_status "All services configured to use GPU 0"
        ;;
    2)
        if [ "$GPU_COUNT" -lt 2 ]; then
            print_error "Dual GPU strategy requires at least 2 GPUs. Found $GPU_COUNT."
            exit 1
        fi
        print_status "Configuring for dual GPU setup..."
        sed -i '/^NVIDIA_VISIBLE_DEVICES=/c\NVIDIA_VISIBLE_DEVICES=0,1' .env
        sed -i '/^# FLOWISE_GPU_DEVICE=/c\FLOWISE_GPU_DEVICE=1' .env
        sed -i '/^# OPEN_WEBUI_GPU_DEVICE=/c\OPEN_WEBUI_GPU_DEVICE=0' .env
        sed -i '/^# N8N_GPU_DEVICE=/c\N8N_GPU_DEVICE=1' .env
        sed -i '/^# QDRANT_GPU_DEVICE=/c\QDRANT_GPU_DEVICE=1' .env
        print_status "GPU 0: Open-WebUI (primary inference)"
        print_status "GPU 1: Flowise, n8n, Qdrant (workflows & embeddings)"
        print_info "Configure Ollama to use GPU 0 with: export CUDA_VISIBLE_DEVICES=0"
        ;;
    3)
        print_status "Configuring for multi-GPU setup..."
        sed -i '/^NVIDIA_VISIBLE_DEVICES=/c\NVIDIA_VISIBLE_DEVICES=all' .env
        
        # Distribute services across available GPUs
        FLOWISE_GPU=$((0 % GPU_COUNT))
        OPEN_WEBUI_GPU=$((1 % GPU_COUNT))
        N8N_GPU=$((2 % GPU_COUNT))
        QDRANT_GPU=$((3 % GPU_COUNT))
        
        sed -i "/^# FLOWISE_GPU_DEVICE=/c\FLOWISE_GPU_DEVICE=$FLOWISE_GPU" .env
        sed -i "/^# OPEN_WEBUI_GPU_DEVICE=/c\OPEN_WEBUI_GPU_DEVICE=$OPEN_WEBUI_GPU" .env
        sed -i "/^# N8N_GPU_DEVICE=/c\N8N_GPU_DEVICE=$N8N_GPU" .env
        sed -i "/^# QDRANT_GPU_DEVICE=/c\QDRANT_GPU_DEVICE=$QDRANT_GPU" .env
        
        print_status "GPU allocation:"
        print_info "  Flowise: GPU $FLOWISE_GPU"
        print_info "  Open-WebUI: GPU $OPEN_WEBUI_GPU"
        print_info "  n8n: GPU $N8N_GPU"
        print_info "  Qdrant: GPU $QDRANT_GPU"
        print_info "Configure Ollama to use all GPUs with: export OLLAMA_NUM_GPU=$GPU_COUNT"
        ;;
    4)
        print_status "Custom GPU configuration..."
        echo "Available GPUs: 0-$((GPU_COUNT-1))"
        
        read -p "Flowise GPU ID (0-$((GPU_COUNT-1))): " FLOWISE_GPU
        read -p "Open-WebUI GPU ID (0-$((GPU_COUNT-1))): " OPEN_WEBUI_GPU
        read -p "n8n GPU ID (0-$((GPU_COUNT-1))): " N8N_GPU
        read -p "Qdrant GPU ID (0-$((GPU_COUNT-1))): " QDRANT_GPU
        
        sed -i "/^# FLOWISE_GPU_DEVICE=/c\FLOWISE_GPU_DEVICE=$FLOWISE_GPU" .env
        sed -i "/^# OPEN_WEBUI_GPU_DEVICE=/c\OPEN_WEBUI_GPU_DEVICE=$OPEN_WEBUI_GPU" .env
        sed -i "/^# N8N_GPU_DEVICE=/c\N8N_GPU_DEVICE=$N8N_GPU" .env
        sed -i "/^# QDRANT_GPU_DEVICE=/c\QDRANT_GPU_DEVICE=$QDRANT_GPU" .env
        
        # Create comma-separated list of unique GPUs
        UNIQUE_GPUS=($(echo "$FLOWISE_GPU $OPEN_WEBUI_GPU $N8N_GPU $QDRANT_GPU" | tr ' ' '\n' | sort -u))
        GPU_LIST=$(IFS=,; echo "${UNIQUE_GPUS[*]}")
        sed -i "/^NVIDIA_VISIBLE_DEVICES=/c\NVIDIA_VISIBLE_DEVICES=$GPU_LIST" .env
        
        print_status "Custom configuration applied"
        ;;
    5)
        print_status "Current GPU configuration:"
        if [ -f .env ]; then
            echo ""
            grep -E "^(NVIDIA_VISIBLE_DEVICES|.*_GPU_DEVICE)" .env | while read line; do
                print_info "  $line"
            done
        else
            print_warning "No .env file found"
        fi
        ;;
    *)
        print_error "Invalid selection"
        exit 1
        ;;
esac

echo ""
print_status "GPU configuration complete!"
echo ""
echo "📝 Next steps:"
echo "1. Review your .env file to verify the settings"
echo "2. Restart your services: docker-compose down && docker-compose up -d"
echo "3. For Ollama, set environment variables before starting:"
echo "   export CUDA_VISIBLE_DEVICES=0,1  # or your preferred GPUs"
echo "   export OLLAMA_NUM_GPU=2          # number of GPUs for Ollama"
echo ""
echo "🔍 Monitor GPU usage with: watch -n 1 nvidia-smi"
echo "📊 Check service logs with: docker-compose logs -f [service_name]"