import chromadb

client = chromadb.PersistentClient(path="chroma_db")

try:
    client.delete_collection(
        name="company_documents"
    )
    print("Old collection deleted.")
except Exception:
    print("Collection did not exist.")

collection = client.get_or_create_collection(
    name="company_documents"
)

print("New empty collection created.")