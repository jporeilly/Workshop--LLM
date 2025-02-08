# WasmEdge CLI Inference Guide

This guide explains how to use WasmEdge for running LLM inference using the command line interface.

## Quick Copy Commands

### Basic Query (128 tokens)
```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm --n-predict 128 --log-enable --prompt "What is a Pentaho Data Integration transformation?"
```

### Technical Query (256 tokens)
```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm --n-predict 256 --log-enable --prompt "Explain the key components of a Kettle transformation and their interactions."
```

### Code Example Query (400 tokens)
```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm --n-predict 400 --log-enable --prompt "Show me a JavaScript example of data transformation using arrays and objects"
```

## Quick Setup Commands

### Create Alias (Single Line)
```bash
echo 'alias llamaquery="wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm"' >> ~/.bashrc && source ~/.bashrc
```

### Create Basic Script
Copy all lines below and paste into terminal:
```bash
cat << 'EOF' > query.sh
#!/bin/bash
DEFAULT_TOKENS=256
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm --n-predict ${2:-$DEFAULT_TOKENS} --log-enable --prompt "$1"
EOF
chmod +x query.sh
```

### Create Advanced Script
Copy all lines below and paste into terminal:
```bash
cat << 'EOF' > advanced-query.sh
#!/bin/bash
DEFAULT_TOKENS=256
CONTEXT_TYPE=${3:-"general"}
PENTAHO_CONTEXT="You are a Pentaho expert. Provide detailed technical answers."
DEBUGGING_CONTEXT="You are a debugging specialist. Focus on troubleshooting steps."
CODE_CONTEXT="You are a programmer. Provide code examples with explanations."
case $CONTEXT_TYPE in
  "pentaho") SYSTEM_CONTEXT="$PENTAHO_CONTEXT" ;;
  "debug") SYSTEM_CONTEXT="$DEBUGGING_CONTEXT" ;;
  "code") SYSTEM_CONTEXT="$CODE_CONTEXT" ;;
  *) SYSTEM_CONTEXT="" ;;
esac
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm --n-predict ${2:-$DEFAULT_TOKENS} --log-enable --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>$SYSTEM_CONTEXT<|eot_id|><|start_header_id|>user<|end_header_id|>$1<|eot_id|>"
EOF
chmod +x advanced-query.sh
```

## Usage Examples

After setting up the alias:
```bash
llamaquery --n-predict 256 --log-enable --prompt "Your question here"
```

Using basic script:
```bash
./query.sh "Your prompt here" 512
```

Using advanced script:
```bash
./advanced-query.sh "How to optimize Pentaho transformations?" 300 pentaho
./advanced-query.sh "Debug slow performance in ETL" 256 debug
./advanced-query.sh "Show me a JavaScript map reduce example" 400 code
```

## Command Parameters Reference

- `--dir .:.` - Sets directory permissions
- `--nn-preload` - Loads neural network model
- `--n-predict` - Number of tokens to generate
- `--log-enable` - Enables logging
- `--prompt` - Your input prompt