# WasmEdge Prompt Template Guide

This guide demonstrates how to structure prompts for specialized use cases using WasmEdge.

## Basic System Role Template

The following template shows how to set up a specialized system role with specific instruction parameters:

```bash
wasmedge --dir .:. \
       --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm \
       --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n You are a Pentaho developer. 
Answer briefly. <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n I need help optimizing a Pentaho Data Integration transformation that's loading data into our data warehouse. <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
```

## Example Templates

### Technical Expert Template
```bash
wasmedge --dir .:. \
       --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm \
       --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a senior data integration specialist with deep expertise in Pentaho.
Provide detailed technical explanations with specific examples.
<|eot_id|><|start_header_id|>user<|end_header_id|>
How can I implement parallel processing in Pentaho transformations?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```

### Code Review Template
```bash
wasmedge --dir .:. \
       --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm \
       --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a code review specialist focusing on ETL best practices.
Analyze code for performance, maintainability, and adherence to standards.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Review this transformation configuration for potential improvements.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```

### Troubleshooting Template
```bash
wasmedge --dir .:. \
       --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm \
       --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a troubleshooting expert specializing in Pentaho Data Integration.
Provide step-by-step debugging approaches and diagnostic strategies.
<|eot_id|><|start_header_id|>user<|end_header_id|>
My transformation is failing with a memory error.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```

### Documentation Template
```bash
wasmedge --dir .:. \
       --nn-preload default:GGML:AUTO:/app/models/llama3-8b-instruct.gguf llama-simple.wasm \
       --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a technical writer specializing in ETL documentation.
Create clear, structured documentation with examples and explanations.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Document the steps to set up a new Pentaho transformation.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
```

## Template Structure Breakdown:

1. **System Role Definition**:
   - `<|begin_of_text|>` - Marks the start of the prompt
   - `<|start_header_id|>system<|end_header_id|>` - Defines system role
   
2. **Instruction Setting**:
   - Sets the role (e.g., "You are a Pentaho developer")
   - Defines response style (e.g., "Answer briefly")

3. **User Query Section**:
   - `<|start_header_id|>user<|end_header_id|>` - Marks user input
   - Followed by the actual query

4. **Assistant Response Setup**:
   - `<|start_header_id|>assistant<|end_header_id|>` - Prepares for assistant response

## Best Practices

1. **Role Specificity**: Be specific about the expertise level and domain knowledge of the system role.
2. **Clear Instructions**: Include specific guidelines about the desired response format and depth.
3. **Context Setting**: Provide relevant context before the actual query.
4. **Response Format**: Structure the template to encourage organized, well-formatted responses.

This template system ensures consistent, role-specific responses with controlled formatting and behavior.