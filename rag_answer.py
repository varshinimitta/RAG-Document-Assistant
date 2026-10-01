from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# Get collection
collection = client.get_collection(
    name="company_documents"
)

# Ask a question
question = input("Ask a question: ")

# Convert question into embedding
question_embedding = model.encode([question])

# Search ChromaDB
results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=1
)

# Get retrieved information
context = results["documents"][0][0]

print("\nRetrieved information:")
print(context)

# Send context + question to Llama
response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "system",
            "content": """You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

Give a short, direct answer.
Do not add information that is not present in the context."""
        },
        {
            "role": "user",
            "content": f"""Context:
{context}

Question:
{question}

Answer:"""
        }
    ]
)

print("\nFinal Answer:")
print(response["message"]["content"])