# 🎓 Technical Interview RAG: AI-Powered Interview Assistant

A high-performance, full-stack **Retrieval-Augmented Generation (RAG)** application engineered for technical interview preparation. 

Query extensive technical study guides (DBMS, Operating Systems, Computer Networks, System Design) and receive structured, interview-ready answers backed by **exact document and page-level citations**—all with sub-second response times.

---

## 🌟 Key Features

* **⚡ Ultra-Fast Groq Inference**: Powered by Groq's high-speed LPU infrastructure with dynamic model discovery (Meta Llama & GPT-OSS models).
* **🔒 100% Free & Private Local Embeddings**: Uses ChromaDB's embedded ONNX model (`all-MiniLM-L6-v2`) running directly on your CPU—zero OpenAI quota limits, zero external embedding costs.
* **📂 Dynamic Multi-Document Uploading**: Drop in new study guides directly via the web UI; documents are automatically chunked, embedded, and added to the vector store on the fly without restarting the server.
* **📖 Verifiable Source Citations**: Every AI response cites the exact source PDF name and page number for rapid verification.
* **💻 Modern Full-Stack Web Interface**: Sleek, responsive, dark-mode ChatGPT-style UI featuring one-click topic starters, markdown parsing, and copyable responses.
* **🚀 FastAPI Backend**: Asynchronous REST API complete with interactive OpenAPI/Swagger documentation at `/docs`.

---

## 🏗️ Architecture

```text
               ┌───────────────────────────────┐
               │    Browser Web Interface      │
               │ (ChatGPT-style Chat & Upload) │
               └───────────────┬───────────────┘
                               │ HTTP / JSON
                               ▼
               ┌───────────────────────────────┐
               │    FastAPI Backend Server     │
               │   (/ask, /upload, /documents) │
               └───────────────┬───────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
   ┌───────────────────────────┐ ┌───────────────────────────┐
   │    Vector Database Engine │ │     LLM Inference Engine  │
   │  ChromaDB + Local ONNX    │ │   Groq Cloud (LPU / Llama)│
   │   (Semantic Search)       │ │ (Structured Interview Ans)│
   └───────────────────────────┘ └───────────────────────────┘
```

---

## 📁 Project Structure

```text
Interview-RAG/
├── backend/
│   ├── documents/              # Stores indexed PDFs (e.g. DBMS_Notes.pdf)
│   ├── static/                 # Frontend assets served by FastAPI
│   │   └── index.html          # Web chat UI
│   ├── app.py                  # FastAPI REST API entry point
│   ├── rag.py                  # Core RAG logic: loading, chunking, retrieval & LLM
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Environment variable template
│   └── .gitignore              # Ignores venv, .env, and vector databases
├── frontend/
│   └── index.html              # Standalone copy of the web chat UI
├── .gitignore                  # Root git ignore
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Interview-RAG.git
cd Interview-RAG/backend
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file inside the `backend/` directory:
```bash
cp .env.example .env
```
Open `.env` and paste your free Groq API key (get one in 30 seconds at [console.groq.com](https://console.groq.com)):
```env
GROQ_API_KEY=gsk_your_actual_key_here
```

### 5. Start the Application
```bash
python app.py
```

* **Web Chat UI**: Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Interactive API Documentation**: Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance async REST API |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) | Lightning-fast ASGI web server |
| **RAG Orchestration** | [LangChain](https://www.langchain.com/) | Document loading, chunking & retrieval pipelines |
| **LLM Inference** | [Groq](https://groq.com/) | High-speed inference using Meta Llama & open models |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) | Local persistent vector database |
| **Embeddings** | `all-MiniLM-L6-v2` via ONNX | Free, local, CPU-based text embeddings |
| **Frontend** | Modern Vanilla JS + CSS | ChatGPT-style dark theme with responsive UI |

---

## 📄 License
MIT License. Free to use for personal interview prep and portfolio projects.
