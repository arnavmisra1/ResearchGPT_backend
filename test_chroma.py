import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- Step 1: Load and chunk the PDF (same as before) ---
file_path = "uploads/Day+11+-+Course+slides.pdf"

loader = PyPDFLoader(file_path)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

print(f"Created {len(chunks)} chunks from the PDF")

# --- Step 2: Load the local embedding model ---
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Step 3: Set up ChromaDB (persistent, saved to disk) ---
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_chunks")

# --- Step 4: Embed each chunk and add to ChromaDB ---
texts = [chunk.page_content for chunk in chunks]
embeddings = embed_model.encode(texts).tolist()
ids = [f"chunk_{i}" for i in range(len(chunks))]
metadatas = [chunk.metadata for chunk in chunks]

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=texts,
    metadatas=metadatas
)

print(f"Added {len(chunks)} chunks to ChromaDB")

# --- Step 5: Test a similarity search ---
query = "What is this document about?"
query_embedding = embed_model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\n--- Top 3 matching chunks for query ---")
for i, doc in enumerate(results["documents"][0]):
    print(f"\nResult {i+1}:")
    print(doc)