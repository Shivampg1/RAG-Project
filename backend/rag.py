import os
import sys
import gc
import json
import urllib.request
import traceback
from pathlib import Path
from dotenv import load_dotenv

# ----------------------------------------------------
# Setup paths reliably (works from any directory)
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DOCS_DIR = BASE_DIR / "documents"
CHROMA_PATH = BASE_DIR / "chroma_db_local"
LOG_PATH = BASE_DIR / "error.log"

load_dotenv(ENV_PATH)

print("=" * 60)
print("🚀 Starting DBMS Interview RAG Assistant (Memory-Optimized)...")
print("=" * 60)

# 1. Check Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    alt_key = os.getenv("OPENAI_API_KEY", "")
    if alt_key.startswith("gsk_"):
        groq_api_key = alt_key
        os.environ["GROQ_API_KEY"] = alt_key

if not groq_api_key:
    print("\n❌ ERROR: 'GROQ_API_KEY' was not found in your environment or .env!")
    print("Ensure GROQ_API_KEY is set in Render Environment Variables.\n")
    sys.exit(1)
else:
    print("✅ Found GROQ_API_KEY")

# 2. Ensure Documents Directory Exists
if not DOCS_DIR.exists():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

# 3. Dynamically discover active models on the user's Groq account
print("🔍 Checking available models on your Groq account...")
def get_best_groq_model(api_key: str) -> str:
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "InterviewRAG/1.0"
            }
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            all_models = [
                m["id"] for m in data.get("data", [])
                if not any(x in m["id"] for x in ["whisper", "guard", "tts", "safetensors", "embed"])
            ]
            
            preferred_keywords = ["gpt-oss", "llama", "qwen", "mistral", "gemma"]
            for kw in preferred_keywords:
                for m in all_models:
                    if kw in m.lower():
                        return m
            
            if all_models:
                return all_models[0]
    except Exception as e:
        print(f"⚠️ Could not fetch dynamic models ({e}), falling back to default.")
    
    return "openai/gpt-oss-20b"

SELECTED_MODEL = get_best_groq_model(groq_api_key)
print(f"🎯 Selected active model: '{SELECTED_MODEL}'")

# 4. Import libraries
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.embeddings import Embeddings
    from langchain_groq import ChatGroq
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
except ImportError as e:
    print(f"\n❌ Missing package: {e}")
    sys.exit(1)

# ----------------------------------------------------
# 100% Free Local Embeddings (Peak Memory Safe)
# ----------------------------------------------------
class LocalEmbeddings(Embeddings):
    """Local embeddings powered by Chroma's built-in ONNX model (all-MiniLM-L6-v2)."""
    def __init__(self):
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raw = self._ef(texts)
        return [[float(val) for val in vec] for vec in raw]

    def embed_query(self, text: str) -> list[float]:
        raw = self._ef([text])
        return [float(val) for val in raw[0]]

try:
    embeddings = LocalEmbeddings()

    def get_vectorstore():
        # If ChromaDB already exists on disk, load it directly
        if CHROMA_PATH.exists() and len(os.listdir(CHROMA_PATH)) > 0:
            print("⚡ Loading existing local ChromaDB (ready instantly)...")
            return Chroma(
                persist_directory=str(CHROMA_PATH),
                embedding_function=embeddings
            )
        
        # Initialize empty Chroma vectorstore
        print("🌱 Initializing local ChromaDB...")
        vs = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=embeddings
        )

        pdf_files = list(DOCS_DIR.glob("*.pdf"))
        if not pdf_files:
            print("ℹ️ No initial PDFs found in documents folder.")
            return vs

        # Stream and batch process initial PDFs to stay well under 512MB RAM
        print(f"⏳ Indexing {len(pdf_files)} initial PDF(s) with memory-safe streaming...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        BATCH_SIZE = 25

        for pdf_file in pdf_files:
            print(f"   Processing {pdf_file.name}...")
            loader = PyPDFLoader(str(pdf_file))
            current_batch = []
            page_count = 0

            # lazy_load streams one page at a time (never keeps 300+ pages in memory!)
            for page in loader.lazy_load():
                page_count += 1
                page_chunks = splitter.split_documents([page])
                current_batch.extend(page_chunks)

                if len(current_batch) >= BATCH_SIZE:
                    vs.add_documents(current_batch)
                    current_batch = []
                    gc.collect()

            if current_batch:
                vs.add_documents(current_batch)
                del current_batch
                gc.collect()

            print(f"   ✅ Finished {pdf_file.name} ({page_count} pages).")

        print("✅ Local Vector database created safely within memory limits!")
        return vs

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model=SELECTED_MODEL,
        temperature=0.2,
        groq_api_key=groq_api_key
    )
    print(f"✅ Groq LLM & Retriever initialized successfully!")

except Exception as e:
    print(f"\n❌ Error during setup: {e}")
    traceback.print_exc()
    sys.exit(1)

# Prompt Template
prompt_template = ChatPromptTemplate.from_template("""
You are an expert technical interviewer and computer science mentor.
Use the following context extracted from the candidate's notes to give a crisp, technically accurate, and well-structured answer.
Include definitions, key points, syntax or examples where relevant, and interview tips.

Context:
{context}

Question:
{question}

Structured Interview Answer:
""")

def ask_question(question: str) -> dict:
    """Answers an interview question using local retrieval and Groq LLM, returning answer and source citations."""
    try:
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        sources = []
        for doc in docs:
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0) + 1
            filename = Path(src).name
            citation = f"{filename} (Page {page})"
            if citation not in sources:
                sources.append(citation)

        messages = prompt_template.format_messages(context=context, question=question)
        response = llm.invoke(messages)
        return {
            "answer": response.content,
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"❌ Error: {e}",
            "sources": []
        }

def add_pdf_to_vectorstore(file_path: str) -> dict:
    """Loads a newly uploaded PDF with streaming and batching to prevent memory spikes."""
    global vectorstore, retriever
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"\n📥 Memory-safe indexing for: {path.name}...")
    loader = PyPDFLoader(str(path))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    BATCH_SIZE = 25
    current_batch = []
    total_pages = 0
    total_chunks = 0

    for page in loader.lazy_load():
        total_pages += 1
        page_chunks = splitter.split_documents([page])
        total_chunks += len(page_chunks)
        current_batch.extend(page_chunks)

        if len(current_batch) >= BATCH_SIZE:
            vectorstore.add_documents(current_batch)
            current_batch = []
            gc.collect()

    if current_batch:
        vectorstore.add_documents(current_batch)
        del current_batch
        gc.collect()

    # Refresh retriever to include new chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"✅ Successfully indexed {total_pages} pages ({total_chunks} chunks) from {path.name}!")

    return {
        "filename": path.name,
        "pages": total_pages,
        "chunks": total_chunks
    }

def get_loaded_documents() -> list[str]:
    """Returns a list of all PDF filenames in the documents directory."""
    if not DOCS_DIR.exists():
        return []
    return sorted([f.name for f in DOCS_DIR.glob("*.pdf")])

# ----------------------------------------------------
# Interactive Terminal Chat Loop
# ----------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(f"🎓 Technical Interview Assistant ({SELECTED_MODEL}) Ready!")
    print("=" * 60 + "\n")
    
    while True:
        try:
            query = input("❓ Question: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye and best of luck with your interviews! 👋\n")
                break
                
            print("\n🤖 Retrieving notes & generating answer...\n")
            res = ask_question(query)
            print(f"💡 Answer:\n{res['answer']}\n")
            if res["sources"]:
                print(f"📖 Sources: {', '.join(res['sources'])}\n")
            print("-" * 60 + "\n")
        except KeyboardInterrupt:
            print("\nExiting. Good luck! 👋")
            break
