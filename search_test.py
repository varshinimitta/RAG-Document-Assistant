from sentence_transformers import SentenceTransformer
import chromadb

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# Get our collection
collection = client.get_collection(
    name="company_documents"
)

# Ask the user for a question
question = input("Ask a question: ")

# Convert question into an embedding
question_embedding = model.encode([question])

# Search ChromaDB
results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=1
)

# Display the relevant information
print("\nRelevant information:")
print(results["documents"][0][0])