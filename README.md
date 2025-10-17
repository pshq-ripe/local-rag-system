                         # Local RAG System for DevOps/SRE

## Table of Contents
- [Introduction and Project Goals](#1-introduction-and-project-goals)
- [System Architecture](#2-system-architecture)
- [Infrastructure Components](#3-infrastructure-components)
- [LLM Model Selection](#4-llm-model-selection)
- [Documentation Indexing](#5-documentation-indexing)
- [MCP Integrations and Extensions](#6-mcp-integrations-and-extensions)
- [Deployment and Operations](#7-deployment-and-operations)
- [Best Practices and Troubleshooting](#8-best-practices-and-troubleshooting)
- [Complete Code Reference](#9-complete-code-reference)

---

## 1. Introduction and Project Goals

### Business Objective
Create a fully local, private RAG (Retrieval-Augmented Generation) system that enables:
- Fast technical answers about DevOps/SRE/Cloud without browsing documentation.
- Data privacy - everything runs locally, zero data sent to external APIs.
- No API costs - unlimited queries without token fees.
- Technical knowledge - indexing O'Reilly books, AWS/Kubernetes/Terraform documentation.
- Tool integration - internet access, Kubernetes, Docker, filesystem.

### Core Principles
- 100% local - no data leaves your computer.
- Production-ready - Docker Compose, health checks, monitoring.
- Scalable - easy to add new documents.
- Flexible - swappable components (models, vector databases).
- Apple Silicon optimized - MLX utilization for maximum performance.

---

## 2. System Architecture

### Component Overview
```markdown
┌─────────────────────────────────────────────────────────────┐
│                       USER / LM STUDIO                      │
│                   (Interface + LLM Model)                   │
└────────────────┬────────────────────────────────────────────┘
│
├──> RAG Queries (port 8000)
│
┌────────────────▼────────────────────────────────────────────┐
│              LANGCHAIN RAG SERVER (Docker)                  │
│  • FastAPI endpoints (/query, /search, /health)             │
│  • LangChain orchestration                                  │
│  • HuggingFace Embeddings (sentence-transformers)           │
│  • Connection pooling                                       │
└────┬──────────────────────────────────┬────────────────────┘
│                                  │
│ Vector Search                    │ LLM Generation
│                                  │
┌────▼─────────────────────┐   ┌────────▼───────────────────┐
│     QDRANT (Docker)      │   │   LM STUDIO LOCAL SERVER   │
│   • 23,389 chunks        │   │   • Qwen3 Coder 30B MLX    │
│   • Similarity search    │   │   • Magistral Small 2509   │
│   • Web Dashboard        │   │   • Tool calling support   │
│   • Port 6333/6334       │   │   • Port 1234              │
└───────────────────────────┘   └────────────────────────────┘
```

### Detailed Query Flow
1. User asks a question in the LM Studio Chat.
2. LM Studio Local Server passes the query to the RAG Server.
3. The RAG Server converts the question into an embedding using sentence-transformers.
4. Qdrant searches for the 3-5 most similar documentation fragments.
5. The RAG Server builds a prompt combining:
   - System instruction
   - Context from Qdrant (retrieved fragments)
   - User's question
6. LM Studio (LLM model) generates an answer using:
   - Context from Qdrant (priority: specific examples, code, facts).
   - Pretrained knowledge (general understanding, syntax, best practices).

### Two Knowledge Sources
| Source | Description | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| Qdrant Retrieved Context | Specific fragments from indexed documents. Exact quotes, examples, code snippets. | Current, precise, verifiable. | Limited to indexed content. |
| Pretrained Model Knowledge | General knowledge about programming, DevOps, clouds, best practices, common patterns. | Broad, structural understanding. | May be outdated (training cutoff date). |

### Technology Stack
- Backend: Python 3.12+, FastAPI, LangChain, Uvicorn
- Databases: Qdrant, Sentence Transformers all-MiniLM-L6-v2 (embeddings)
- Containerization: Docker & Docker Compose
- LLM: LM Studio, MLX quantized models

---

## 3. Infrastructure Components

### 3.1. Qdrant Vector Database
- **Role**: Store and search vector representations of documentation.
- **Specifications**:
  - Version: `qdrant/qdrant:latest`
  - Ports: 6333 (HTTP API), 6334 (gRPC)
  - Storage: Docker volume `qdrant_storage`
  - Collection: `devops_docs` (23,389 chunks)
  - Vector dimensions: 384 (all-MiniLM-L6-v2)
  - Health Check: `http://localhost:6333` (every 10s)
  - Resource Usage:
    - CPU: ~0.5-1 core
    - RAM: ~200-500 MB
    - Disk: ~2-5 GB

### 3.2. LangChain RAG Server
- **Role**: RAG pipeline orchestration, API endpoint for queries.
- **Specifications**:
  - Framework: FastAPI + LangChain
  - Port: 8000
  - Base Image: `python:3.12-slim`
  - API Endpoints:
    - GET `/`: Service information
    - GET `/health`: Detailed health check
    - GET `/config`: Current configuration
    - POST `/query`: Main RAG query endpoint
    - POST `/search`: Direct search without an LLM

### 3.3. LM Studio Local Server
- **Role**: Host LLM models, inference, tool calling.
- **Specifications**:
  - Port: 1234 (OpenAI-compatible API)
  - Platform: Apple Silicon M4 Pro, 48 GB RAM
  - Installed Models:
    - Qwen3 Coder 30B MLX 6BIT (~25 GB)
    - Magistral Small 2509 MLX 5BIT (~17 GB)

---

## 4. LLM Model Selection

### 4.1. Selection Criteria for DevOps/SRE
- Technical accuracy: Precision in Terraform, K8s, AWS.
- Code generation: HCL, YAML, Docker Compose.
- Tool calling support: Integration with MCP servers.
- M4 performance: MLX optimization.

### 4.2. Model Comparison

| Model | Specifications | Strengths | Weaknesses | Best for... |
|-------|----------------|-----------|------------|-------------|
| Qwen3 Coder 30B MLX (6-bit) ⭐⭐⭐⭐⭐ | 30B, 25GB, 30-40 tok/s | Best for code/infra, great Terraform/K8s knowledge, fast on MLX | Requires ~35 GB RAM, slower than smaller models | Generating Terraform modules, debugging K8s manifests, code review. |
| Magistral Small 2509 MLX (5-bit) ⭐⭐⭐⭐⭐ | 22B, 17GB, 40-50 tok/s | Excellent reasoning, lighter and faster than Qwen3, good at technical writing | Slightly weaker in pure code generation | Architectural decisions, complex problem solving, best practice recommendations. |

### 4.3. Deployment Recommendations
- **For 48 GB RAM**:
  - Primary: Qwen3 Coder 30B (for code/infrastructure)
  - Secondary: Magistral Small 2509 (for reasoning/decisions)
- **For 32 GB RAM**:
  - Primary: Magistral Small 2509 (universal)
- **For 64+ GB RAM**:
  - Premium: Qwen3 Coder 30B 8BIT (max quality)

---

## 5. Documentation Indexing

### 5.1. Document Preparation
- **Supported formats**: PDF, TXT, Markdown, HTML.
- **Folder structure**:
documents/
├── devops/
├── sre/
└── cloud/

- **Recommended sources**: O'Reilly books, official documentation, internal company documentation.

### 5.2. Indexing Process
1. **Loading**: Read documents from folders (PyPDFLoader).
2. **Chunking**:
 - Chunk size: 1000 characters
 - Overlap: 200 characters (to preserve context)
 - Result: 23,389 chunks from 8,677 pages.
3. **Embedding Generation**:
 - Model: `sentence-transformers/all-MiniLM-L6-v2`
 - Dimensions: 384
4. **Storing in Qdrant**:
 - Collection: `devops_docs`
 - Distance metric: Cosine Similarity
 - Indexing time: ~30-60 minutes.

---

## 6. MCP Integrations and Extensions

### 6.1. Model Context Protocol (MCP) Overview
- **What is MCP**: A protocol created by Anthropic that enables LLMs to access external tools.
- **Architecture**: LLM → MCP Server → External Service/API

### 6.2. Web Search Integration
- **Implementation**: `mrkrsl/web-search-mcp`
- **Available Tools**:
- `full-web-search`: Comprehensive search with full content extraction.
- `get-web-search-summaries`: Quick search with snippets.
- `get-single-web-page-content`: Extract content from a specific URL.

### 6.3. Kubernetes MCP Server
- **Implementation**: `containers/kubernetes-mcp-server`
- **Key Features**:
- Pod management (list, logs, exec).
- CRUD for any K8s resource (Deployments, Services, etc.).
- Helm operations (install, list, uninstall).
- **Security Modes**:
- Read-only: View only.
- Disable destructive: View and create, but no updates/deletes.
- Full access: Full permissions (for dev environments).

---

## 7. Deployment and Operations

### 7.1. Docker Compose Setup
- **Services**:
- `qdrant`: The vector database.
- `langchain-server`: The RAG server.
- LM Studio: Runs natively on the host (outside of Docker).
- **Project Structure**:
```markdown
local-mcp/
├── docker-compose.yaml        # Main orchestration file
├── Dockerfile                 # RAG server container image
├── requirements.txt           # Python dependencies
├── rag_server.py              # FastAPI RAG server
├── index_documents.py         # Document indexing script
├── Makefile                   # Convenience commands
├── mcp.json                   # MCP configuration for LM Studio
|
├── documents/                 # Source documents for indexing
|   ├── devops/
|   ├── sre/
|   └── cloud/
|
└── logs/                      # Application logs
```

### 7.2. Networking
- **Docker Network**: `rag-network` (bridge type).
- **Port Mapping**:
- 6333 → Qdrant HTTP API
- 8000 → RAG Server API
- 1234 → LM Studio (host)
- **Host Access**: `host.docker.internal` allows the RAG server to communicate with LM Studio.

### 7.3. Health Checks & Monitoring
- **Qdrant Health**: `GET http://localhost:6333`
- **RAG Server Health**: `GET http://localhost:8000/health` (returns the status of connections and collections).

---

## 8. Best Practices and Troubleshooting

### 8.1. Model Selection Strategy
- Code generation: Qwen3 Coder 30B
- Architectural decisions: Magistral Small 2509
- Quick queries: Magistral Small 2509
- Debugging: Qwen3 Coder 30B

### 8.2. RAG Query Optimization
- `max_results (k)`: Default is 3. Increasing it improves context but slows down the response.
- `temperature`: Default is 0.7. For DevOps tasks, 0.5-0.7 is recommended for more predictable answers.

### 8.3. Common Issues and Solutions
- **Problem**: RAG Server won't start.
- **Solution**: Check Qdrant logs (`docker compose logs qdrant`), verify the network, and restart the container.
- **Problem**: Collection not found (404).
- **Solution**: Run the indexing script (`make index`), verify the collection exists via the Qdrant API, and restart the server.
- **Problem**: LM Studio connection error.
- **Solution**: Ensure the local server in LM Studio is running and a model is loaded. Test the connection (`curl http://localhost:1234/v1/models`).

---

## 9. Complete Code Reference
All code files are located in this repository. Refer to the [Project Structure](#71-docker-compose-setup) for details.