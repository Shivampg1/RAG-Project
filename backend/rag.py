import os
import sys
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
PDF_PATH = DOCS_DIR / "DBMS_Notes.pdf"
CHROMA_PATH = BASE_DIR / "chroma_db_local"
LOG_PATH = BASE_DIR / "error.log"

load_dotenv(ENV_PATH)

print("=" * 60)
print("🚀 Starting DBMS Interview RAG Assistant (Groq + Local Embeddings)...")
print("=" * 60)

# 1. Check Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    alt_key = os.getenv("OPENAI_API_KEY", "")
    if alt_key.startswith("gsk_"):
        groq_api_key = alt_key
        os.environ["GROQ_API_KEY"] = alt_key

if not groq_api_key:
    print("\n❌ ERROR: 'GROQ_API_KEY' was not found in your .env file!")
    print(f"Please check your .env file at: {ENV_PATH}")
    print("Ensure it contains: GROQ_API_KEY=gsk_...\n")
    sys.exit(1)
else:
    print("✅ Found GROQ_API_KEY in .env")

# 2. Check PDF exists
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
    print("👉 Run: pip install langchain-groq python-multipart\n")
    sys.exit(1)

# ----------------------------------------------------
# 100% Free Local Embeddings
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
        if CHROMA_PATH.exists() and len(os.listdir(CHROMA_PATH)) > 0:
            print("⚡ Loading existing local ChromaDB (ready instantly)...")
            return Chroma(
                persist_directory=str(CHROMA_PATH),
                embedding_function=embeddings
            )
        
        # Initial index if documents exist
        pdf_files = list(DOCS_DIR.glob("*.pdf"))
        if not pdf_files:
            print("ℹ️ No initial PDFs found in documents folder.")
            return Chroma(
                persist_directory=str(CHROMA_PATH),
                embedding_function=embeddings
            )

        print(f"⏳ Reading {len(pdf_files)} initial PDF(s) and creating local ChromaDB...")
        all_chunks = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)
        
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_PATH)
        )
        print("✅ Local Vector database created!")
        return vectorstore

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
        
        # Extract unique sources with page numbers
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
    """Loads a newly uploaded PDF, splits into chunks, and adds to Chroma vector database."""
    global vectorstore, retriever
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"\n📥 Indexing newly uploaded document: {path.name}...")
    loader = PyPDFLoader(str(path))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    vectorstore.add_documents(chunks)
    # Refresh retriever to include new documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"✅ Successfully indexed {len(documents)} pages ({len(chunks)} chunks) from {path.name}!")

    return {
        "filename": path.name,
        "pages": len(documents),
        "chunks": len(chunks)
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
