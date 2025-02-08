# WasmEdge CLI Inference Guide

This guide explains how to use WasmEdge for running LLM inference using the command line interface.

## Basic Command Structure

The basic command structure follows this pattern:

```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm --n-predict <tokens> --log-enable --prompt "<your prompt>"
```

### Command Parameters Explained:
- `--dir .:.` - Sets the directory permissions
- `--nn-preload` - Loads the neural network model
- `--n-predict` - Number of tokens to generate
- `--log-enable` - Enables logging
- `--prompt` - Your input prompt

## Examples

### Basic Query
```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm --n-predict 128 --log-enable --prompt "What is a Pentaho Data Integration transformation?"
```
This example runs a basic query with 128 tokens about Pentaho Data Integration.

### Technical Documentation Query
```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm --n-predict 256 --log-enable --prompt "Explain the key components of a Kettle transformation and their interactions."
```
This example requests a detailed technical explanation with 256 tokens.

### Code Example Query
```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm --n-predict 400 --log-enable --prompt "Show me a JavaScript example of data transformation using arrays and objects"
```
This example requests a longer response (400 tokens) for a code explanation.

### Troubleshooting Query
```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm --n-predict 300 --log-enable --prompt "How to debug performance issues in a Pentaho ETL job that's running slowly?"
```
This example focuses on problem-solving with a moderate token length.

### Best Practices Query
```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm --n-predict 350 --log-enable --prompt "What are the best practices for designing efficient Pentaho Data Integration workflows?"
```
This example requests guidelines and recommendations with an extended token count.

## Simplified Usage

### Alias Setup
You can create an alias for easier use:

```bash
echo 'alias llamaquery="wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm"' >> ~/.bashrc
source ~/.bashrc
```

After setting up the alias, you can use:
```bash
llamaquery --n-predict 256 --log-enable --prompt "Your question here"
```

### Shell Script
Create a reusable shell script for even more flexibility:

```bash
#!/bin/bash
DEFAULT_TOKENS=256

wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf \
  llama-simple.wasm \
  --n-predict ${2:-$DEFAULT_TOKENS} \
  --log-enable \
  --prompt "$1"
```

Usage:
```bash
./query.sh "Your prompt here" 512
```
This script allows you to specify both the prompt and token count, with a default of 256 tokens if not specified.

### Advanced Shell Script with Templates
```bash
#!/bin/bash
DEFAULT_TOKENS=256
CONTEXT_TYPE=${3:-"general"}

# Predefined contexts
PENTAHO_CONTEXT="You are a Pentaho expert. Provide detailed technical answers."
DEBUGGING_CONTEXT="You are a debugging specialist. Focus on troubleshooting steps."
CODE_CONTEXT="You are a programmer. Provide code examples with explanations."

case $CONTEXT_TYPE in
  "pentaho")
    SYSTEM_CONTEXT="$PENTAHO_CONTEXT"
    ;;
  "debug")
    SYSTEM_CONTEXT="$DEBUGGING_CONTEXT"
    ;;
  "code")
    SYSTEM_CONTEXT="$CODE_CONTEXT"
    ;;
  *)
    SYSTEM_CONTEXT=""
    ;;
esac

wasmedge --dir .:. \
  --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf \
  llama-simple.wasm \
  --n-predict ${2:-$DEFAULT_TOKENS} \
  --log-enable \
  --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
$SYSTEM_CONTEXT
<|eot_id|><|start_header_id|>user<|end_header_id|>
$1
<|eot_id|>"
```

Usage:
```bash
./advanced-query.sh "How to optimize Pentaho transformations?" 300 pentaho
./advanced-query.sh "Debug slow performance in ETL" 256 debug
./advanced-query.sh "Show me a JavaScript map reduce example" 400 code
```