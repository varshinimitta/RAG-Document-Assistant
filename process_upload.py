import chromadb
from sentence_transformers import SentenceTransformer

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# Get our collection
collection = client.get_collection(
    name="company_documents"
)

# Load the embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Read the uploaded file
with open("uploads/test.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split the text into chunks
chunks = []

chunk_size = 500

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

print("Number of chunks:", len(chunks))

# Create embeddings
embeddings = model.encode(chunks).tolist()

# Store chunks in ChromaDB
for i in range(len(chunks)):

    collection.add(
        documents=[chunks[i]],
        embeddings=[embeddings[i]],
        ids=[f"uploaded_doc_{i}"]
    )

print("Uploaded document successfully stored in ChromaDB!")