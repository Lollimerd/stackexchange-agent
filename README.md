
-----

# StackOverflow GraphRAG Chatbot

This project is a sophisticated Graph-based Retrieval-Augmented Generation (GraphRAG) application. It leverages a Neo4j graph database to store and query StackOverflow data, a FastAPI backend to handle logic, and a multi-featured Streamlit UI for user interaction. The system can answer questions by retrieving relevant context from the knowledge graph and synthesizing answers using a local LLM hosted via Ollama.

## ✨ Features

  * **Advanced GraphRAG Pipeline**: Moves beyond simple vector search by using an `EnsembleRetriever` to pull context from `Question`, `Answer`, `User`, and `Tag` nodes in the graph.
  * **Rich Context Retrieval**: A custom Cypher query fetches not just the relevant question but also its associated answers, tags, and user details, providing rich, interconnected context to the LLM.
  * **Streaming Responses**: The backend streams the LLM's response directly to the UI for a real-time, ChatGPT-like experience.
  * **Visible Agent Thoughts**: The LLM is prompted to "think" before answering. This thought process is captured and displayed in a collapsible expander in the UI, providing transparency into the agent's reasoning.
  * **Interactive Multi-Chat UI**: The Streamlit frontend supports multiple, independent chat sessions. Users can create, delete, and switch between chats, with history saved for each session.
  * **Dynamic Data Ingestion**: A "Stackoverflow Loader" tab in the UI allows users to pull data directly from the Stack Exchange API by tag or by top-voted questions and load it into the Neo4j database.
  * **Configuration Display**: The UI fetches and displays key configuration details (like the Ollama model and Neo4j connection info) from the backend.

## 🏗️ Architecture

The application is composed of three main components that work together:

1.  **Neo4j Database**: The knowledge graph that stores StackOverflow data (Questions, Answers, Users, Tags) and their relationships. Vector indexes are created on nodes for efficient similarity search.
2.  **FastAPI Backend (`llmz.py`)**:
      * Exposes a `/stream-ask` endpoint that receives a user's question.
      * Embeds the question and uses a LangChain `EnsembleRetriever` to perform a hybrid search across multiple Neo4j vector indexes.
      * Executes a detailed Cypher query to retrieve a rich subgraph of context around the matched questions.
      * Formats the retrieved context and the user's question into a prompt for the LLM.
      * Streams the generated response back to the client, using special tags (`<|THINK_START|>`, `<|THINK_END|>`) to delineate the model's thought process from the final answer.
      * Provides a `/api/v1/config` endpoint for the frontend.
3.  **Streamlit Frontend (`streamlit_UI.py`)**:
      * Provides the user interface for chatting.
      * Manages multiple chat sessions, including history.
      * Sends user queries to the FastAPI backend and processes the streamed response.
      * Parses the special "think" tags to separate and display the agent's reasoning in an expander.
      * Includes a data loader tab that allows users to populate the Neo4j database directly from the Stack Exchange API.

## 🛠️ Tech Stack

  * **Backend**: FastAPI, LangChain, Uvicorn
  * **Frontend**: Streamlit
  * **Database**: Neo4j (via `langchain-neo4j`)
  * **LLM & Embeddings**: Ollama (e.g., `qwen3:1.7b`, `bge-m3`)
  * **Core Libraries**: `requests`, `python-dotenv`

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.9+**: Ensure you have a recent version of Python installed.
2.  **Neo4j Database**: A running Neo4j instance (AuraDB, Docker, or local install).
3.  **Ollama**: Ollama must be installed and running.
      * Pull the required models:
        ```bash
        ollama pull qwen3:1.7b
        ollama pull bge-m3
        ```

### 1\. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2\. Install Dependencies

It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

*(Note: A `requirements.txt` file should be created from the imports in the provided scripts.)*

### 3\. Configure Environment Variables

Create a `.env` file in the root of the project by copying the `.env.example` template.

**.env.example**

```env
# Neo4j Connection Details
NEO4J_URL="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASS="your_neo4j_password"

# Ollama Connection
OLLAMA_BASE_URL="http://localhost:11434"
```

Fill in the `.env` file with your actual Neo4j and Ollama details.

### 4\. Run the Application

The provided `start.sh` script will run both the backend and frontend services concurrently.

```bash
chmod +x start.sh
./start.sh
```

Alternatively, you can run them in separate terminals:

  * **Terminal 1: Start the FastAPI Backend**
    ```bash
    uvicorn llmz:app --host 0.0.0.0 --port 8080
    ```
  * **Terminal 2: Start the Streamlit Frontend**
    ```bash
    streamlit run streamlit_UI.py
    ```

### 5\. Load Data and Chat

1.  Open your browser and navigate to the Streamlit URL (usually `http://localhost:8501`).
2.  Go to the **"Stackoverflow loader"** tab.
3.  Enter a tag (e.g., `python`, `neo4j`) and the number of pages to import.
4.  Click **"Import"**. This will fetch data from the Stack Exchange API, generate embeddings, and populate your Neo4j database. This may take a few minutes.
5.  Once the import is complete, navigate back to the **"Custom Bot"** tab and start asking questions\!

## 📂 File Structure

```
.
├── llmz.py             # FastAPI backend server with the core GraphRAG logic.
├── streamlit_UI.py     # Main Streamlit frontend application with multi-chat UI.
├── loader.py           # Data ingestion logic for the StackOverflow loader tab.
├── utils.py            # Utility functions for formatting and DB setup.
├── qbot.py             # A simpler, single-chat version of the Streamlit UI.
├── start.sh            # Convenience script to run backend and frontend.
├── .env                # (You create this) Secret keys and configuration.
└── requirements.txt    # (You create this) Python package dependencies.
```

## 🧠 How It Works: A Deeper Dive

#### 1\. Data Ingestion (`loader.py`)

When you use the loader UI, the application calls the Stack Exchange API. For each question and its answers, it:

1.  Concatenates the title and body to form a single text block.
2.  Uses the `OllamaEmbeddings` class (`bge-m3` model) to generate a vector embedding for the text.
3.  Executes a single, powerful Cypher query (`UNWIND $data AS q...`) to efficiently batch-create all the `Question`, `Answer`, `User`, and `Tag` nodes and their relationships (`:ANSWERS`, `:TAGGED`, etc.) in Neo4j.

#### 2\. Retrieval (`llmz.py`)

This is the core of the "GraphRAG" process.

1.  **Ensemble Retrieval**: Instead of relying on one vector index, we initialize four `Neo4jVector.from_existing_graph` retrievers, one for each main node type. The `EnsembleRetriever` combines the results from all four, providing a more diverse set of initial candidate nodes.
2.  **Custom Cypher Injection**: The `retrieval_query` is the most critical part. When a retriever finds a candidate node (e.g., a `Question` node via vector search), this query doesn't just return that node's text. Instead, it uses the node as an entry point to explore the graph. It traverses relationships to gather the asker's details, all associated tags, and a collection of all answers with their providers' details.
3.  **Structured Output**: The query formats this rich subgraph information into a structured `metadata` JSON object, which is attached to the LangChain `Document`.

#### 3\. Generation & Streaming (`llmz.py` & `streamlit_UI.py`)

1.  The retrieved `Documents` (with their rich metadata) are formatted into a single context string by `format_docs_with_metadata`.
2.  This context is inserted into a prompt that explicitly instructs the LLM to first think step-by-step inside `<think></think>` tags and then provide the final answer.
3.  The FastAPI backend uses `astream` to get a token-by-token stream from the LLM.
4.  As chunks of text arrive, a buffer is used to detect the special `<|THINK_START|>` and `<|THINK_END|>` tags.
5.  The Streamlit frontend consumes this stream, directing content to either the "Agent Thoughts" expander or the main answer placeholder based on whether the stream is currently inside a "thinking" block. This creates a seamless and transparent user experience.