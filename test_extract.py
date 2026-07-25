from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


file_path = "uploads/Day+11+-+Course+slides.pdf"


loader = PyPDFLoader(file_path)
pages = loader.load()

print(f"Number of pages extracted: {len(pages)}")
print("--- First page content preview ---")
print(pages[0].page_content[:500])

    
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(pages)

print(f"\nNumber of chunks created: {len(chunks)}")
print("--- First chunk ---")
print(chunks[0].page_content)
print("--- Chunk metadata (tracks which page it came from) ---")
print(chunks[0].metadata)