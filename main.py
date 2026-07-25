from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import json
from fastapi.responses import FileResponse
from typing import List
import sqlite3
from datetime import datetime

# --- NEW: SQLite setup, runs once at startup ---
DB_PATH = "annotations.db"
MAX_FILE_SIZE_MB = 20


load_dotenv()

# --- NEW: imports for the RAG pipeline ---
import chromadb
from sentence_transformers import SentenceTransformer


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            highlighted_text TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL
        )
    """)
    try:
        conn.execute("ALTER TABLE annotations ADD COLUMN position_data TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists, ignore

    conn.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            filename TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            topics TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

def generate_summary(chunks):
    # Use a sample of chunks (first ~10) to keep the prompt reasonably sized,
    # rather than sending the entire document's text
    sample_text = "\n\n".join(chunk.page_content for chunk in chunks[:10])

    prompt = f"""Based on the following document excerpt, provide:
1. A 2-3 sentence summary of what this document is about
2. A list of 4-6 key topics covered

Respond in this exact JSON format, with no other text:
{{"summary": "...", "topics": ["topic1", "topic2", ...]}}

Document excerpt:
{sample_text}"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        timeout=15
    )

    raw = response.choices[0].message.content

    # the model might wrap JSON in markdown code fences despite instructions; strip those if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)
    return parsed["summary"], parsed["topics"]

init_db()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # keep for local development
        os.getenv("FRONTEND_URL", "")  # NEW: will be set as an env var on Render
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnnotationCreate(BaseModel):
    filename: str
    page_number: str
    highlighted_text: str
    note: str = ""
    position_data: str = ""
    
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    filename: str
    history: List[Message] = []

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- NEW: load embedding model + set up ChromaDB ONCE, at startup ---
# This runs a single time when uvicorn starts, not on every request.
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_chunks")
print("Embedding model and ChromaDB ready.")

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

@app.post("/annotations")
def create_annotation(annotation: AnnotationCreate):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO annotations (filename, page_number, highlighted_text, note, position_data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (annotation.filename, annotation.page_number, annotation.highlighted_text, annotation.note, annotation.position_data, datetime.now().isoformat())
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return {"id": new_id, "message": "Annotation saved"}

@app.get("/annotations/{filename}")
def get_annotations(filename: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    rows = conn.execute(
        "SELECT * FROM annotations WHERE filename = ? ORDER BY page_number, created_at",
        (filename,)
    ).fetchall()
    conn.close()

    return [dict(row) for row in rows]

@app.delete("/annotations/{annotation_id}")
def delete_annotation(annotation_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
    conn.commit()
    conn.close()
    return {"message": "Annotation deleted"}

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/debug/chunks")
def debug_chunks():
    all_data = collection.get()
    sources = set(m["source"] for m in all_data["metadatas"])
    return {"unique_sources_stored": list(sources)}

@app.get("/pdf/{filename}")
def get_pdf(filename: str):
    file_path = f"{UPLOAD_DIR}/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type="application/pdf")

@app.get("/summary/{filename}")
def get_summary(filename: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM summaries WHERE filename = ?", (filename,)).fetchone()
    conn.close()

    if not row:
        return {"summary": None, "topics": []}

    return {"summary": row["summary"], "topics": json.loads(row["topics"])}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Check 1: file extension
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the file first (we need it saved to check size and parse it)
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    content = await file.read()

    # Check 2: file size
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File too large ({size_mb:.1f}MB). Max size is {MAX_FILE_SIZE_MB}MB.")

    with open(file_path, "wb") as f:
        f.write(content)

    # Check 3: can we actually parse it as a PDF?
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    except Exception as e:
        os.remove(file_path)  # clean up the bad file
        raise HTTPException(status_code=400, detail="Could not read this PDF. It may be corrupted or password-protected.")

    # Check 4: does it actually contain extractable text?
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    if len(chunks) == 0:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="No readable text found in this PDF. It may be a scanned image without OCR.")

    texts = [chunk.page_content for chunk in chunks]
    embeddings = embed_model.encode(texts).tolist()

    ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [chunk.metadata for chunk in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    # NEW: generate and store a summary for this document
    try:
        summary, topics = generate_summary(chunks)
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO summaries (filename, summary, topics) VALUES (?, ?, ?)",
            (file.filename, summary, json.dumps(topics))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Summary generation failed: {e}")  # don't block upload if this fails

    return {
        "filename": file.filename,
        "message": "Upload successful",
        "chunks_created": len(chunks)
    }

@app.post("/chat")
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    query_embedding = embed_model.encode(request.question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"source": {"$eq": f"uploads/{request.filename}"}}
    )
    retrieved_chunks = results["documents"][0]

    if len(retrieved_chunks) == 0:
        def empty_stream():
            msg = {'type': 'answer_chunk', 'text': "I couldn't find any relevant content in this document to answer that question."}
            yield f"data: {json.dumps(msg)}\n\n"
            done = {'type': 'done', 'sources': []}
            yield f"data: {json.dumps(done)}\n\n"

        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    context = "\n\n".join(retrieved_chunks)
    sources = [r["page"] for r in results["metadatas"][0]]

    # NEW: system instructions, sent once, separate from conversation history
    system_message = {
        "role": "system",
        "content": f"""You are answering questions about a document. Use the context below to answer clearly and directly, in your own words. If a question refers back to something discussed earlier in the conversation, use that context too.

If the answer isn't in the provided context, say "I don't have enough information in the document to answer that."

Context from the document:
{context}"""
    }

    # NEW: convert incoming history into the format Groq expects
    history_messages = [{"role": m.role, "content": m.content} for m in request.history]

    # NEW: full message list = system instructions + past turns + new question
    messages = [system_message] + history_messages + [{"role": "user", "content": request.question}]

    def generate():
        try:
            stream = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,  # CHANGED: full conversation, not just one prompt string
                stream=True,
                timeout=15
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'type': 'answer_chunk', 'text': delta})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': 'The AI service is currently unavailable. Please try again.'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.delete("/document/{filename}")
def delete_document(filename: str):
    # Find all chunk IDs belonging to this document
    source_path = f"uploads/{filename}"
    all_data = collection.get(where={"source": {"$eq": source_path}})

    if len(all_data["ids"]) == 0:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Delete those chunks from ChromaDB
    collection.delete(ids=all_data["ids"])

    # Also delete the actual file from disk
    file_path = f"{UPLOAD_DIR}/{filename}"
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"message": f"Deleted {filename}", "chunks_removed": len(all_data["ids"])}

