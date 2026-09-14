from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import uvicorn

# Import RAG functions from rag.py
from rag import ask_question, add_pdf_to_vectorstore, get_loaded_documents, DOCS_DIR

BASE_DIR = Path(__file__).resolve().parent
STATIC_INDEX = BASE_DIR / "static" / "index.html"

# 1. Initialize FastAPI application
app = FastAPI(
    title="Multi-Document Interview RAG API",
    description="Backend API for querying interview notes using RAG and Groq LLM with dynamic document uploads",
    version="1.1.0"
)

# 2. Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define Request & Response Schemas
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str] = []

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    pages: int
    chunks: int

# 4. Endpoints
@app.get("/")
def serve_ui():
    """Serves the interactive ChatGPT-style web UI."""
    if STATIC_INDEX.exists():
        return FileResponse(STATIC_INDEX)
    return {
        "status": "healthy",
        "message": "DBMS Interview RAG Backend is up and running!"
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint to verify the server is running."""
    return {
        "status": "healthy",
        "message": "Interview RAG Backend is up and running!"
    }

@app.get("/documents")
def list_documents():
    """Returns the list of all currently indexed PDF documents."""
    return {
        "documents": get_loaded_documents()
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Uploads a new PDF document, chunks it, and adds it to the active vector database."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files (.pdf) are supported.")
    
    destination = DOCS_DIR / file.filename
    try:
        # Save file to documents directory
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Index into Chroma vector database
        info = add_pdf_to_vectorstore(str(destination))
        return UploadResponse(
            status="success",
            message=f"Successfully indexed '{file.filename}' into the vector store.",
            filename=info["filename"],
            pages=info["pages"],
            chunks=info["chunks"]
        )
    except Exception as e:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/ask", response_model=AnswerResponse)
def handle_ask_question(payload: QuestionRequest):
    """Takes a question, retrieves relevant notes from indexed PDFs, and returns an answer with citations."""
    user_query = payload.question.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        result = ask_question(user_query)
        return AnswerResponse(
            question=user_query,
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {str(e)}")

# ----------------------------------------------------
# Run server directly with `python app.py`
# ----------------------------------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting FastAPI server on port {port}...")
    print(f"💻 Web Chat UI available at: http://127.0.0.1:{port}")
    print(f"📖 Interactive Swagger API Docs at: http://127.0.0.1:{port}/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
