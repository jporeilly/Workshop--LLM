## RAG - Chainlit

### Setup
1. Clone the repo and navigate to the folder:
    ```
    git clone https://github.com/jporeilly/Workshop--LLM.git
    cd  Workshop--LLM/Playground/chainlit
    ls
    ```

2. Ensure `uv` is installed:
    - Installation: [UV Getting Started](https://docs.astral.sh/uv/getting-started/installation/)

3. Install the required packages - creates virtual env:
    ```
    cd  Workshop--LLM/Playground/chainlit
    uv sync
    ```

4. Qdrant Docker container. Set qdrant url in the `.env` file.

    Qdrant documentation: [Qdrant Documentation](https://qdrant.tech/documentation/quickstart/)
    
    - Rename `.env.example` to `.env`
    - Provide your env variables inside it as shown below.
    ```
    QDRANT_URL_LOCALHOST="xxxxx"
    ```

    ```
    docker pull qdrant/qdrant
    docker run --name qdrant -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
    ```

5. Run the chainlit app
    ```
    cd  Workshop--LLM/Playground/chainlit
    uv run setup-rag.py
    uv run chainlit run rag-chainlit-deepseek.py
    ```