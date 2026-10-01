from sentence_transformers import SentenceTransformer
import chromadb

# 1. Read the document
file_path = "data/documents/company.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()


# 2. Split the document into chunks
chunk_size = 150
chunks = []

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)


# 3. Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)


# 4. Create ChromaDB
client = chromadb.PersistentClient(path="chroma_db")


# 5. Create a collection
collection = client.get_or_create_collection(
    name="company_documents"
)


# 6. Store chunks and embeddings
collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings.tolist()
)


print("Number of chunks:", len(chunks))
print("Documents successfully stored in ChromaDB!")