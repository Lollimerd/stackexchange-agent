# StackExchange GraphRAG Agent

> A private RAG-powered technical Q&A agent backed by a Neo4j knowledge graph and local LLMs.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.13+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53+-red)

---

## 🚀 Quick Start

```bash
# 1. Pull required models
ollama pull qwen3.5:4b
ollama pull snowflake-arctic-embed2

# 2. Configure environment
cp .env.example .env
# Edit .env with your Neo4j credentials

# 3. Run everything
./run.sh
```

**Access the app:** `http://localhost:8510`

---

## 📋 What It Does

This agent answers technical questions by:

1. **Retrieving** relevant context from a Neo4j graph (Questions, Answers, Users, Tags)
2. **Reranking** results for relevance
3. **Generating** detailed answers with visible reasoning

Data is sourced from the **Stack Exchange API** and embedded locally—nothing leaves your machine.

---

## 🏗️ Architecture

```mermaid
graph TD
    User([User]) --> Streamlit[Streamlit Frontend<br/>:8510]
    Streamlit <-->|SSE Stream| FastAPI[FastAPI Backend<br/>:8000]
    
    FastAPI --> Agent[GraphRAG Agent]
    Agent --> Tools[graph_rag_tool]
    Tools --> Retriever[Ensemble Retriever<br/>Hybrid Search]
    Retriever <--> Neo4j[(Neo4j Graph DB<br/>:7474, :7687)]
    Agent --> LLM[Ollama LLM<br/>:11430]
    
    Loader[Data Loader] -.->|Fetch| SE[Stack Exchange API]
    Loader -->|Embed + Store| Neo4j
    
    style Streamlit fill:#ff4b4b,color:#fff
    style FastAPI fill:#009688,color:#fff
    style Neo4j fill:#018bff,color:#fff
    style LLM fill:#ffa500,color:#000
```

### Components

| Service | Port | Description |
|---------|------|-------------|
| **Streamlit Frontend** | 8510 | Multi-chat UI with graph explorer & dashboard |
| **FastAPI Backend** | 8000 | Agent orchestration, streaming, ingestion |
| **Neo4j** | 7474, 7687 | Knowledge graph with vector indexes |
| **Ollama** | 11430 | Local LLM inference |

---

## 🧠 Models Used

| Model | Purpose | Context |
|-------|---------|---------|
| `qwen3.5:4b` | Main chat & reasoning | 128K |
| `qwen3.5:0.8b` | Chat history summarization | 40K |
| `snowflake-arctic-embed2` | Embeddings | 8K |
| `BAAI/bge-reranker-base` | Document reranking | - |

---

## ✨ Features

### Core
- **GraphRAG Pipeline** — Hybrid search across Question, Answer, User, and Tag nodes
- **Streaming Responses** — Real-time token-by-token generation
- **Visible Reasoning** — Agent thoughts displayed in expandable sections
- **Multi-Chat Support** — Independent sessions with persistent history

### Data & Visualization
- **StackExchange Loader** — Import by tag or top-voted questions
- **Graph Explorer** — Interactive Neo4j visualization
- **Analytics Dashboard** — Import history & database statistics

### Privacy
- **100% Local** — All models run via Ollama
- **No External Calls** — Except Stack Exchange API for imports

---

## 🛠️ Tech Stack

```
Backend:  FastAPI, LangChain, Uvicorn, Neo4j
Frontend: Streamlit, Mermaid
LLM:      Ollama (qwen3.5, snowflake-arctic-embed2)
Infra:    Docker Compose, Neo4j (APOC + GDS plugins)
```

---

## 📦 Project Structure

```
.
├── backend/
│   ├── agent/              # LangChain agent & middleware
│   ├── app/
│   │   └── backend.py      # FastAPI server & endpoints
│   ├── tools/              # graph_rag_tool implementation
│   ├── utils/              # DB functions, memory management
│   └── setup/              # Model & graph initialization
│
├── frontend/
│   ├── web.py              # Main chat interface
│   ├── pages/
│   │   ├── loader.py       # Data ingestion UI
│   │   ├── neo4j_explorer.py
│   │   └── dashboard.py
│   └── utils/
│
├── docker-compose.yml      # Service orchestration
├── pyproject.toml          # Python dependencies
└── run.sh                  # Dev launcher
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```bash
# Neo4j
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here

# Stack Exchange API (optional, for higher rate limits)
STACKEXCHANGE_API_KEY=your_key_here
```

### Ollama Setup

```bash
# Required models
ollama pull qwen3.5:4b
ollama pull snowflake-arctic-embed2

# Optional (for summarization)
ollama pull qwen3.5:0.8b
```

---

## 🎯 API Endpoints

### Chat & Agent
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agent/ask` | POST | Stream agent responses (SSE) |
| `/api/v1/chat/{id}` | GET/DELETE | Fetch/delete chat history |
| `/api/v1/users` | GET | List all users |

### Data Ingestion
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ingest` | POST | Import StackExchange data |
| `/api/v1/ingest/record` | POST/PUT/DELETE | Manage import sessions |

### Analytics
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/stats/summary` | GET | Database statistics |
| `/api/v1/stats/history` | GET | Import history |
| `/api/v1/graph/search` | GET | Search nodes |
| `/api/v1/graph/sample` | POST | Graph sample for visualization |

---

## 🔍 How GraphRAG Works

### 1. Data Ingestion
```
Stack Exchange API → Embed (snowflake-arctic-embed2) → Neo4j
```
Questions, answers, users, and tags are stored as nodes with relationships:
- `(:User)-[:ASKED]->(:Question)`
- `(:Question)<-[:ANSWERS]-(:Answer)`
- `(:Question)-[:TAGGED]->(:Tag)`
- `(:Answer)<-[:PROVIDED]-(:User)`

### 2. Retrieval
```
User Question → Embed → Hybrid Search (Vector + BM25) → Rerank → Context
```
- **4 vector indexes** (Question, Answer, User, Tag)
- **EnsembleRetriever** combines results
- **Cross-encoder reranking** for relevance

### 3. Generation
```
Context + Prompt → qwen3.5:4b → Streamed Response
```
Agent prompted to:
1. Think step-by-step (captured separately)
2. Generate Mermaid diagrams when helpful
3. Provide technically precise answers

---

## 🐳 Docker Deployment

```bash
docker compose up -d
```

**Services:**
- `ollama` — LLM inference with GPU passthrough
- `graphDB` — Neo4j with APOC & GDS plugins
- `backend-server` — FastAPI with NVIDIA GPU access
- `frontend` — Streamlit UI

**Volumes:**
- `ollama_data` — Persistent model storage
- `neo4j_stackoverflow_data` — Graph database

---

## 📊 Neo4j Schema

### Node Labels
- `Question` — title, body, score, creation_date, embedding
- `Answer` — body, score, is_accepted, embedding
- `User` — display_name, reputation
- `Tag` — name
- `Session` — chat session metadata
- `Message` — conversation history
- `ImportLog` — ingestion tracking

### Relationships
```
(User)-[:ASKED]->(Question)
(Question)<-[:ANSWERS]-(Answer)
(Question)-[:TAGGED]->(Tag)
(Answer)<-[:PROVIDED]-(User)
(Session)-[:HAS_MESSAGE]->(Message)
```

---

## 🧪 Development

### Run Backend (dev)
```bash
cd backend
python -m app.backend
```

### Run Frontend (dev)
```bash
cd frontend
streamlit run web.py
```

### Run Tests
```bash
pytest
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Submit a PR

---

## 📝 License

MIT License — see LICENSE for details.

---

## 🙏 Acknowledgments

- **Stack Exchange** — Data API
- **Neo4j** — Graph database
- **Ollama** — Local LLM runtime
- **LangChain** — Agent framework
- **Streamlit** — Rapid UI development

---

**Built with ❤️ using local LLMs**
