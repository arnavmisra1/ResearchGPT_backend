import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

# --- Part 1: Test Groq chat ---
groq_key = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=groq_key,
    base_url="https://api.groq.com/openai/v1"
)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "Say hello in exactly 5 words."}
    ]
)

print("--- Groq chat response ---")
print(response.choices[0].message.content)

# --- Part 2: Test local embeddings ---
print("\n--- Loading local embedding model (first run downloads it, be patient) ---")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

text = "This is a test sentence for embeddings."
vector = embed_model.encode(text)

print(f"Embedding generated. Vector length: {len(vector)}")
print(f"First 5 values: {vector[:5]}")