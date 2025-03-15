# AI Stack Setup Guide

This guide provides comprehensive instructions for setting up a complete AI development stack using Docker Compose. The stack includes workflow automation, databases, large language model serving, vector database, and visualization tools.

## Components

This AI stack includes the following components:

1. **n8n** - Workflow automation platform
2. **PostgreSQL** - Relational database for n8n
3. **SQLite** - Lightweight database for Flowise record management
4. **Ollama** - Local LLM serving (CPU and GPU options)
5. **Qdrant** - Vector database for similarity search
6. **Open-WebUI** - Web interface for Ollama models
7. **Flowise** - Low-code AI automation platform

## Prerequisites

- Docker and Docker Compose installed
- Basic understanding of Docker containers
- For GPU support: NVIDIA GPU with appropriate drivers and nvidia-docker

## Directory Structure

Create the following directory structure:

```
GenAI/
├── docker-compose.yml
├── .env
├── n8n/
│   └── backup/
│       ├── credentials/
│       └── workflows/
└── shared/
```

## Setup Instructions

### 1. Create Directories

```bash
mkdir -p GenAI/n8n/backup/credentials
mkdir -p GenAI/n8n/backup/workflows
mkdir -p GenAI/shared
```

### 2. Create Docker Compose File

Create a file named `docker-compose.yml` in the `GenAI` directory with the following content:

```yaml
# Docker Compose configuration for an AI development stack
# This setup includes n8n (workflow automation), Postgres, Ollama (LLM serving), 
# Qdrant (vector database), Open-WebUI, and Flowise

# Define volumes for persistent storage
volumes:
  # Storage for n8n data (workflows, executions, etc.)
  n8n_storage:
  # Storage for PostgreSQL database files
  postgres_storage:
  # Storage for Ollama models and configuration
  ollama_storage:
  # Storage for Qdrant vector database
  qdrant_storage:
  # Storage for Open-WebUI configuration and data
  open-webui:
  # Storage for Flowise configuration and data
  flowise:

# Define networks for container communication
networks:
  # Main network that all services will connect to
  demo:

# Common service configuration for n8n containers (using YAML anchor)
x-n8n: &service-n8n
  image: n8nio/n8n:latest
  networks: ['demo']
  environment:
    # Database configuration for n8n
    - DB_TYPE=postgresdb
    - DB_POSTGRESDB_HOST=postgres
    - DB_POSTGRESDB_USER=${POSTGRES_USER}  # Uses environment variable
    - DB_POSTGRESDB_PASSWORD=${POSTGRES_PASSWORD}  # Uses environment variable
    # n8n specific configuration
    - N8N_RUNNERS_ENABLED=true  # Enable workflow runners for performance
    - N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true  # Security measure for settings file
    - N8N_DIAGNOSTICS_ENABLED=false  # Disable sending diagnostics data
    - N8N_PERSONALIZATION_ENABLED=false  # Disable personalization features
    # Security keys (must be provided in environment or .env file)
    - N8N_ENCRYPTION_KEY
    - N8N_USER_MANAGEMENT_JWT_SECRET
  links:
    - postgres  # Creates a DNS entry for the postgres service

# Common configuration for Ollama service (using YAML anchor)
x-ollama: &service-ollama
  image: ollama/ollama:latest
  networks: ['demo']
  restart: unless-stopped
  ports:
    - 11435:11434  # Map to a different external port to avoid conflicts
  volumes:
    - ollama_storage:/root/.ollama  # Persist Ollama models and data

# Common configuration for initializing Ollama with models (using YAML anchor)
x-init-ollama: &init-ollama
  image: ollama/ollama:latest
  networks: ['demo']
  entrypoint: /bin/sh
  volumes:
    - ollama_storage:/root/.ollama  # Share the same volume as the Ollama service
  command:
    - "-c"
    # Waits for Ollama to start, then pulls the specified models
    # Using internal container port 11434 for communication within the Docker network
    - "sleep 10; OLLAMA_HOST=ollama:11434 ollama pull llama3.1; OLLAMA_HOST=ollama:11434 ollama pull nomic-embed-text"

# Define all services
services:
  # Flowise - a low-code AI automation platform
  flowise:
    image: flowiseai/flowise
    networks: ['demo']
    restart: unless-stopped
    container_name: flowise
    environment:
        - PORT=3001
    ports:
        - 3001:3001  # Expose Flowise UI on port 3001
    extra_hosts:
      - "host.docker.internal:host-gateway"  # Allow access to host machine services    
    volumes:
        - ~/.flowise:/root/.flowise  # Store Flowise data in user's home directory
    entrypoint: /bin/sh -c "sleep 3; flowise start"  # Wait before starting to ensure dependencies are ready

  # Open-WebUI - a web interface for Ollama models
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    networks: ['demo']
    restart: unless-stopped
    container_name: open-webui
    ports:
      - "3000:8080"  # Map container port 8080 to host port 3000
    extra_hosts:
      - "host.docker.internal:host-gateway"  # Allow access to host machine services
    volumes:
      - open-webui:/app/backend/data  # Persist Open-WebUI data

  # PostgreSQL database - used by n8n for workflow storage
  postgres:
    image: postgres:16-alpine
    networks: ['demo']
    restart: unless-stopped
    ports:
      - 5432:5432  # Expose PostgreSQL on default port
    environment:
      - POSTGRES_USER  # Username from environment variable
      - POSTGRES_PASSWORD  # Password from environment variable
      - POSTGRES_DB  # Database name from environment variable
    volumes:
      - postgres_storage:/var/lib/postgresql/data  # Persist database files
    healthcheck:
      # Check if PostgreSQL is ready to accept connections
      test: ['CMD-SHELL', 'pg_isready -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB}']
      interval: 5s
      timeout: 5s
      retries: 10

  # n8n-import - one-time container to import workflows and credentials
  n8n-import:
    <<: *service-n8n  # Inherit configuration from service-n8n anchor
    container_name: n8n-import
    entrypoint: /bin/sh
    command:
      - "-c"
      # Import credentials and workflows from backup directory
      - "n8n import:credentials --separate --input=/backup/credentials && n8n import:workflow --separate --input=/backup/workflows"
    volumes:
      - ./GenAI/n8n/backup:/backup  # Mount local backup directory
    depends_on:
      postgres:
        condition: service_healthy  # Wait until PostgreSQL is healthy

  # n8n - workflow automation platform
  n8n:
    <<: *service-n8n  # Inherit configuration from service-n8n anchor
    container_name: n8n
    restart: unless-stopped
    ports:
      - 5678:5678  # Expose n8n UI on port 5678
    volumes:
      - n8n_storage:/home/node/.n8n  # Persist n8n data
      - ./GenAI/n8n/backup:/backup  # Mount local backup directory
      - ./GenAI/shared:/data/shared  # Mount shared directory for data exchange
    depends_on:
      postgres:
        condition: service_healthy  # Wait until PostgreSQL is healthy
      n8n-import:
        condition: service_completed_successfully  # Wait until import is complete

  # Qdrant - vector database for similarity search
  qdrant:
    image: qdrant/qdrant
    container_name: qdrant
    networks: ['demo']
    restart: unless-stopped
    ports:
      - 6333:6333  # Expose Qdrant API on default port
    volumes:
      - qdrant_storage:/qdrant/storage  # Persist vector database files

  # Ollama for CPU - activated with "cpu" profile
  ollama-cpu:
    profiles: ["cpu"]  # Only start with "cpu" profile
    <<: *service-ollama  # Inherit configuration from service-ollama anchor
    container_name: ollama  # Using a single name for service discovery

  # Ollama for GPU (NVIDIA) - activated with "gpu-nvidia" profile
  ollama-gpu:
    profiles: ["gpu-nvidia"]  # Only start with "gpu-nvidia" profile
    <<: *service-ollama  # Inherit configuration from service-ollama anchor
    container_name: ollama  # Using a single name for service discovery
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]  # Request GPU access for the container

  # Initialize Ollama with models (CPU version)
  ollama-pull-llama-cpu:
    profiles: ["cpu"]  # Only start with "cpu" profile
    <<: *init-ollama  # Inherit configuration from init-ollama anchor
    container_name: ollama-pull-llama-cpu
    depends_on:
      - ollama-cpu  # Wait for Ollama CPU service to start

  # Initialize Ollama with models (GPU version)
  ollama-pull-llama-gpu:
    profiles: ["gpu-nvidia"]  # Only start with "gpu-nvidia" profile
    <<: *init-ollama  # Inherit configuration from init-ollama anchor
    container_name: ollama-pull-llama-gpu
    depends_on:
      - ollama-gpu  # Wait for Ollama GPU service to start
```

### 3. Create Environment File

Create a file named `.env` in the `GenAI` directory with the following content:

```
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_DB=n8n
N8N_ENCRYPTION_KEY=your_strong_encryption_key
N8N_USER_MANAGEMENT_JWT_SECRET=your_strong_jwt_secret
```

Replace the placeholder values with your own secure credentials.

## Starting the Stack

### CPU-only Mode

To start the stack in CPU-only mode:

```bash
cd GenAI
docker-compose --profile cpu up -d
```

### NVIDIA GPU Mode

To start the stack with NVIDIA GPU support:

```bash
cd GenAI
docker-compose --profile gpu-nvidia up -d
```

## Accessing Services

Once the stack is running, you can access the various services at these URLs:

- **n8n**: http://localhost:5678
- **Flowise**: http://localhost:3001
- **Open-WebUI**: http://localhost:3000
- **Qdrant API**: http://localhost:6333
- **Ollama API**: http://localhost:11435
- **PostgreSQL**: localhost:5432
- **SQLite**: Data persisted in sqlite_storage volume for Flowise

## Building AI Workflows

The stack is designed for building AI-powered workflows:

1. **Document Processing Workflow Example**:
   - Upload documents to a shared folder
   - Trigger n8n workflow on new document
   - Extract text using n8n
   - Generate embeddings using Ollama's nomic-embed-text model
   - Store document and embeddings in PostgreSQL or Qdrant
   - Index vectors in Qdrant for similarity search
   - Build a search interface in Flowise

2. **AI Assistant Workflow Example**:
   - Create chatbot interface in Open-WebUI
   - Connect to Ollama's llama3.1 model
   - Enable RAG (Retrieval-Augmented Generation) by connecting to Qdrant
   - Create n8n workflow to log and analyze conversations

## Troubleshooting

### Common Issues

1. **Container fails to start**:
   - Check logs: `docker logs <container_name>`
   - Verify environment variables are set correctly
   - Ensure ports are not already in use

2. **Network connectivity issues**:
   - Ensure all containers are on the 'demo' network
   - Check container health: `docker ps -a`

3. **Database connection issues**:
   - Verify credentials in .env file
   - Check database logs: `docker logs postgres`

4. **GPU not detected**:
   - Run `nvidia-smi` to verify GPU is accessible
   - Ensure nvidia-docker is properly installed

## Maintenance

### Backup

To backup your data:

```bash
# Backup PostgreSQL
docker exec -t postgres pg_dump -U ${POSTGRES_USER} ${POSTGRES_DB} > pg_backup.sql

# Backup SQLite for Flowise
docker cp sqlite-flowise:/data/flowise.sqlite ./flowise_backup.sqlite
```

### Upgrading Components

To upgrade a specific component:

```bash
# Pull latest image
docker-compose pull <service_name>

# Restart service
docker-compose --profile <profile_name> up -d <service_name>
```

## Advanced Configuration

### Custom Ollama Models

To add custom models to Ollama:

1. Edit the `x-init-ollama` section in docker-compose.yml:

```yaml
command:
  - "-c"
  - "sleep 10; OLLAMA_HOST=ollama:11434 ollama pull llama3.1; OLLAMA_HOST=ollama:11434 ollama pull nomic-embed-text; OLLAMA_HOST=ollama:11434 ollama pull your-custom-model"
```

2. Restart the Ollama service:

```bash
docker-compose --profile cpu restart ollama-cpu
# or
docker-compose --profile gpu-nvidia restart ollama-gpu
```

### Scaling Vector Database

For larger vector databases, you can adjust Qdrant's configuration:

1. Create a `qdrant_config.yaml` file in the GenAI directory
2. Mount it to the Qdrant container by adding this to the volumes:

```yaml
- ./GenAI/qdrant_config.yaml:/qdrant/config/production.yaml
```

## Security Considerations

1. **Environment Variables**: Never commit .env files to version control
2. **Network Security**: The stack is configured for local development only
3. **Authentication**: Add proper authentication for production use
4. **Data Privacy**: Be careful with sensitive data in document storage

## Resources

- [n8n Documentation](https://docs.n8n.io/)
- [Ollama Documentation](https://ollama.ai/documentation)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Flowise Documentation](https://docs.flowiseai.com/)