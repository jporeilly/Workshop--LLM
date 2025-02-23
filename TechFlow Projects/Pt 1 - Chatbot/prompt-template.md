# WasmEdge Prompt Template Guide

This guide provides copy-paste ready templates for different use cases.

## Quick Copy Templates

### Basic Technical Query
```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/qwen-3b-instruct.gguf llama-simple.wasm --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a Pentaho developer. Answer briefly.<|eot_id|><|start_header_id|>user<|end_header_id|>I need help optimizing a Pentaho Data Integration transformation.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```

### Technical Expert Query
```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/qwen-3b-instruct.gguf llama-simple.wasm --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a senior data integration specialist with deep expertise in Pentaho. Provide detailed technical explanations with specific examples.<|eot_id|><|start_header_id|>user<|end_header_id|>How can I implement parallel processing in Pentaho transformations?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```

### Code Review Query
```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/qwen-3b-instruct.gguf llama-simple.wasm --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a code review specialist focusing on ETL best practices. Analyze code for performance, maintainability, and adherence to standards.<|eot_id|><|start_header_id|>user<|end_header_id|>Review this transformation configuration for potential improvements.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```

### Troubleshooting Query
```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/qwen-3b-instruct.gguf llama-simple.wasm --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a troubleshooting expert specializing in Pentaho Data Integration. Provide step-by-step debugging approaches.<|eot_id|><|start_header_id|>user<|end_header_id|>My transformation is failing with a memory error.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```

## Create Template Script
Copy all lines below and paste into terminal:
```bash
cat << 'EOF' > template-query.sh
#!/bin/bash
ROLE=${2:-"general"}
case $ROLE in
  "expert") 
    SYSTEM="You are a senior data integration specialist with deep expertise in Pentaho. Provide detailed technical explanations with specific examples."
    ;;
  "review")
    SYSTEM="You are a code review specialist focusing on ETL best practices. Analyze code for performance, maintainability, and adherence to standards."
    ;;
  "debug")
    SYSTEM="You are a troubleshooting expert specializing in Pentaho Data Integration. Provide step-by-step debugging approaches."
    ;;
  *)
    SYSTEM="You are a Pentaho developer. Answer briefly."
    ;;
esac
wasmedge --dir .:. --nn-preload default:GGML:AUTO:/app/models/qwen-3b-instruct.gguf llama-simple.wasm --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>$SYSTEM<|eot_id|><|start_header_id|>user<|end_header_id|>$1<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
EOF
chmod +x template-query.sh
```

## Usage Examples

Using template script:
```bash
./template-query.sh "How to optimize this transformation?" expert
./template-query.sh "Review my ETL job configuration" review
./template-query.sh "Debug memory issues in my job" debug
./template-query.sh "What is a kettle transformation?" general
```

## Template Structure

All templates follow this pattern:
```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
[System Role and Instructions]
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
[User Query]
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```