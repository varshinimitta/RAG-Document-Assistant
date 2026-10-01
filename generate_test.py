import chromadb
import requests

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# Get our collection
collection = client.get_collection(
    name="company_documents"
)

# Ask the user a question
question = input("Ask a question: ")

# Search ChromaDB
results = collection.query(
    query_texts=[question],
    n_results=1
)

# Get the relevant document
context = results["documents"][0][0]

print("\nRetrieved information:")
print(context)

# Create a prompt for Llama
prompt = f"""
You are a helpful question-answering assistant.

Answer the question using ONLY the information provided below.

Information:
{context}

Question:
{question}

Answer:
"""

# Send prompt to Ollama
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }
)

# Get the answer
answer = response.json()["response"]

print("\nFinal Answer:")
print(answer)