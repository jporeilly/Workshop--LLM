## Llamaparse 

### Setup
1. Clone the repo and navigate to the folder:
    ```
    git clone https://github.com/jporeilly/Workshop--LLM.git
    cd  Workshop--LLM/Playground/agentic-rag
    ls
    ```

2. Ensure `uv` is installed:
    - Installation: [UV Getting Started](https://docs.astral.sh/uv/getting-started/installation/)

3. Qdrant Docker container.

    Set qdrant url in the `.env` file.
     [openai](https://platform.openai.com/settings/organization/api-keys). 
     [Qdrant Cloud](https://cloud.qdrant.io/)

    Qdrant documentation: [Qdrant Documentation](https://qdrant.tech/documentation/quickstart/)

    - Rename `.env.example` to `.env`
    - Provide your env variables inside it as shown below.

    [openai](https://platform.openai.com/settings/organization/api-keys)
    [Qdrant Cloud](https://cloud.qdrant.io/) - if using cloud

    ```
    OPENAI_API_KEY="xxxxx"
    QDRANT_URL="xxxxx"
    QDRANT_API_KEY="xxxxx"
    QDRANT_URL_LOCALHOST="xxxxx"
    ```

    ```
    docker pull qdrant/qdrant
    docker run --name qdrant -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
    ```

4. Install the required packages - creates virtual env:
    ```
    cd  Workshop--LLM/Playground/agentic-rag
    uv sync
    ```

5. 

## Agentic RAG Advantages

Agentic Retrieval-Augmented Generation (RAG) combines intelligent agents with retrieval-augmented generation to enhance data retrieval and decision-making processes. Here’s a simplified overview of its key advantages:
- Improved Reasoning for Better Responses: Agentic RAG enhances the reasoning capabilities of AI systems, leading to more accurate and contextually relevant responses.    ￼
- Smart Tool Selection Based on Queries: It enables intelligent agents to choose the most appropriate tools or data sources, such as knowledge bases or search engines, based on the context of a query.
- Integrated Memory for Context Awareness: By integrating memory, Agentic RAG allows AI systems to remember and utilize previous interactions, improving context awareness and continuity in conversations. ￼
- Effective Task Planning and Breakdown: The system can decompose complex tasks into manageable sub-tasks, enabling better planning and execution by the AI agents. ￼
- Seamless Integration with Various Data Sources: Agentic RAG offers flexibility in connecting with diverse data sources, including PDFs, websites, CSV files, and documents, enhancing its versatility across different applications. ￼

These features collectively make Agentic RAG a powerful approach for optimizing data retrieval and decision-making in AI systems.