#!/bin/bash

# AI Stack Setup Script
# This script helps you set up the AI development environment

set -e

echo "🚀 Setting up AI Development Stack..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    print_warning "Ollama is not installed. Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Check GPU setup
print_status "Checking GPU configuration..."
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
    
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | tr -d '\n')
    print_status "Found $GPU_COUNT GPU(s)"
    
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo ""
        echo "💡 Multiple GPUs detected! You can optimize performance by:"
        echo "   - Assigning specific GPUs to different services"
        echo "   - Setting FLOWISE_GPU_DEVICE=0, OPEN_WEBUI_GPU_DEVICE=1, etc."
        echo "   - Configuring Ollama to use specific GPUs with OLLAMA_NUM_GPU"
        echo ""
    fi
else
    print_warning "nvidia-smi not found. Make sure NVIDIA drivers are installed."
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p flowise-uploads
mkdir -p n8n-uploads
mkdir -p open-webui-uploads
mkdir -p postgres-init
mkdir -p qdrant-config
mkdir -p redis-data

# Set proper permissions
chmod 755 flowise-uploads
chmod 755 n8n-uploads
chmod 755 open-webui-uploads
chmod 755 postgres-init
chmod 755 qdrant-config
chmod 755 redis-data

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file..."
    cp .env.template .env
    
    # Generate secure random keys
    print_status "Generating secure random keys..."
    
    # Generate 32-character encryption key for n8n (needed for secure credential storage)
    ENCRYPTION_KEY=$(openssl rand -hex 16)
    sed -i "s|your_32_character_encryption_key_here|$ENCRYPTION_KEY|g" .env
    
    # Generate Flowise secret
    FLOWISE_SECRET=$(openssl rand -hex 32)
    sed -i "s|your_flowise_secret_key_here|$FLOWISE_SECRET|g" .env
    
    # Generate secure password (alphanumeric only to avoid sed issues)
    POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    sed -i "s|your_secure_password_here|$POSTGRES_PASSWORD|g" .env
    
    FLOWISE_PASSWORD=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)
    sed -i "s|your_flowise_password_here|$FLOWISE_PASSWORD|g" .env
    
    print_status "Generated secure random keys and passwords in .env file"
    print_warning "Please review and customize the .env file as needed"
else
    print_status ".env file already exists"
fi

# Create external volume for Flowise
print_status "Creating external volume for Flowise..."
docker volume create genai_flowise || true

# Pull latest images
print_status "Pulling latest Docker images..."
docker-compose pull

# Start Ollama service if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    print_status "Starting Ollama service..."
    
    # Configure Ollama for multiple GPUs if available
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | tr -d '\n')
        if [ "$GPU_COUNT" -gt 1 ]; then
            export OLLAMA_NUM_GPU=$GPU_COUNT
            print_status "Configuring Ollama to use $GPU_COUNT GPUs"
        fi
    fi
    
    ollama serve &
    sleep 5
fi

# Pull recommended models
print_status "Pulling recommended Ollama models..."
ollama pull llama3.1:latest || print_warning "Failed to pull llama3.1:latest"
ollama pull nomic-embed-text:latest || print_warning "Failed to pull nomic-embed-text:latest"
ollama pull llama3.2:latest || print_warning "Failed to pull llama3.2:latest"

# Optional: Pull additional useful models
read -p "Do you want to pull additional models? (codellama, mistral, phi3) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Pulling additional models..."
    ollama pull codellama:latest || print_warning "Failed to pull codellama:latest"
    ollama pull mistral:latest || print_warning "Failed to pull mistral:latest"
    ollama pull phi3:latest || print_warning "Failed to pull phi3:latest"
fi

# Start the stack
print_status "Starting AI development stack..."
docker-compose up -d

# Wait for services to start
print_status "Waiting for services to start..."
sleep 30

# Check service health
print_status "Checking service health..."
echo "🔍 Service Status:"
echo "  - Flowise: http://localhost:3001"
echo "  - Open-WebUI: http://localhost:8090"
echo "  - n8n: http://localhost:5678"
echo "  - Qdrant: http://localhost:6333"
echo "  - PostgreSQL: localhost:5435"
echo "  - Redis: localhost:6380"

# Test connections
print_status "Testing service connections..."
curl -f http://localhost:3001/api/v1/ping > /dev/null 2>&1 && echo "✅ Flowise is running" || echo "❌ Flowise is not responding"
curl -f http://localhost:8090/health > /dev/null 2>&1 && echo "✅ Open-WebUI is running" || echo "❌ Open-WebUI is not responding"
curl -f http://localhost:5678/healthz > /dev/null 2>&1 && echo "✅ n8n is running" || echo "❌ n8n is not responding"
curl -f http://localhost:6333/health > /dev/null 2>&1 && echo "✅ Qdrant is running" || echo "❌ Qdrant is not responding"

print_status "Setup complete! 🎉"
echo ""
echo "📝 Next steps:"
echo "1. Access the services using the URLs above"
echo "2. Configure your workflows in n8n"
echo "3. Set up your AI flows in Flowise"
echo "4. Start chatting with your models in Open-WebUI"
echo ""
echo "📋 Useful commands:"
echo "  - View logs: docker-compose logs -f [service_name]"
echo "  - Stop stack: docker-compose down"
echo "  - Restart stack: docker-compose restart"
echo "  - Update images: docker-compose pull && docker-compose up -d"