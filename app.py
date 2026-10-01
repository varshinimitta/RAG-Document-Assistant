from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import os
from pypdf import PdfReader

app = Flask(__name__)


# -----------------------------
# Load embedding model
# -----------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# Connect to ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="company_documents"
)


# -----------------------------
# Home page
# -----------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# -----------------------------
# Ask Question API
# -----------------------------

@app.route(
    "/api/ask",
    methods=["POST"]
)
def ask_question():

    data = request.get_json()

    if not data:

        return {
            "error": "Invalid request."
        }, 400


    question = data.get(
        "question"
    )


    if not question:

        return {
            "error": "Question is required."
        }, 400


    # -----------------------------
    # Convert question into embedding
    # -----------------------------

    question_embedding = model.encode(
        [question]
    )


    # -----------------------------
    # Search ChromaDB
    # Retrieve top 3 chunks
    # -----------------------------

    results = collection.query(

        query_embeddings=
            question_embedding.tolist(),

        n_results=3

    )


    # -----------------------------
    # Get retrieved documents
    # -----------------------------

    retrieved_documents = (
        results["documents"][0]
    )


    print(
        "\nRetrieved Context:"
    )


    for document in retrieved_documents:

        print(document)


    # -----------------------------
    # Combine retrieved chunks
    # -----------------------------

    context = "\n\n".join(
        retrieved_documents
    )


    # -----------------------------
    # Get source filename
    # -----------------------------

    source = "Unknown document"


    if results["metadatas"][0]:

        for metadata in results["metadatas"][0]:

            if metadata:

                source = metadata.get(
                    "source",
                    "Unknown document"
                )

                break


    # -----------------------------
    # Ask Llama
    # -----------------------------

    response = ollama.chat(

        model="llama3.2:3b",

        messages=[

            {
                "role": "system",

                "content": """
You are a document question-answering assistant.

Your job is to answer the user's question
using ONLY the provided context.

Rules:

1. Use only information from the context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Do not add information that is not in the context.
5. Give a short and accurate answer.
6. If the context directly contains the answer,
   preserve the important information from
   the original sentence.
7. Do not unnecessarily shorten the answer.
8. If the answer cannot be found in the context,
   say exactly:

Sorry, I could not find the answer in the uploaded document.
"""
            },

            {
                "role": "user",

                "content": f"""
Context:

{context}


Question:

{question}


Answer:
"""
            }

        ]

    )


    # -----------------------------
    # Get Llama answer
    # -----------------------------

    answer = response[
        "message"
    ][
        "content"
    ].strip()


    # -----------------------------
    # Preserve complete context
    # -----------------------------
    # If Llama returns only a shortened
    # part of a retrieved document,
    # return the complete retrieved text.

    for document in retrieved_documents:

        if answer.lower() in document.lower():

            answer = document

            break


    # -----------------------------
    # Clean common Llama formatting
    # -----------------------------

    answer = answer.replace(
        "Answer:",
        ""
    ).strip()


    # -----------------------------
    # Return result
    # -----------------------------

    return {

        "question": question,

        "answer": answer,

        "source": source

    }


# -----------------------------
# Upload + Process Document
# -----------------------------

@app.route(
    "/api/upload",
    methods=["POST"]
)
def upload_file():


    # -----------------------------
    # Check if file exists
    # -----------------------------

    if "file" not in request.files:

        return {
            "error": "No file uploaded."
        }, 400


    file = request.files["file"]


    # -----------------------------
    # Check if file was selected
    # -----------------------------

    if file.filename == "":

        return {
            "error": "No file selected."
        }, 400


    # -----------------------------
    # Check file type
    # -----------------------------

    allowed_extensions = [

        ".txt",

        ".pdf"

    ]


    file_extension = os.path.splitext(
        file.filename
    )[1].lower()


    if file_extension not in allowed_extensions:

        return {

            "error":
                "Only TXT and PDF files are allowed."

        }, 400


    # -----------------------------
    # Create uploads folder
    # -----------------------------

    os.makedirs(
        "uploads",
        exist_ok=True
    )


    # -----------------------------
    # Save uploaded file
    # -----------------------------

    file_path = os.path.join(

        "uploads",

        file.filename

    )


    file.save(
        file_path
    )


    # -----------------------------
    # Read document
    # -----------------------------

    if file.filename.lower().endswith(
        ".pdf"
    ):

        reader = PdfReader(
            file_path
        )


        text = ""


        for page in reader.pages:

            page_text = page.extract_text()


            if page_text:

                text += (
                    page_text + "\n"
                )


    else:

        with open(

            file_path,

            "r",

            encoding="utf-8"

        ) as f:

            text = f.read()


    # -----------------------------
    # Check extracted text
    # -----------------------------

    if not text.strip():

        return {

            "error":
                "No readable text found in the document."

        }, 400


    # -----------------------------
    # Split document into chunks
    # -----------------------------

    paragraphs = text.split(
        "\n"
    )


    chunks = []


    for paragraph in paragraphs:

        paragraph = paragraph.strip()


        if paragraph:

            chunks.append(
                paragraph
            )


    # -----------------------------
    # Check chunks
    # -----------------------------

    if not chunks:

        return {

            "error":
                "No usable text chunks found."

        }, 400


    # -----------------------------
    # Create embeddings
    # -----------------------------

    embeddings = model.encode(
        chunks
    ).tolist()


    # -----------------------------
    # Store in ChromaDB
    # -----------------------------

    for i in range(
        len(chunks)
    ):

        collection.add(

            documents=[
                chunks[i]
            ],

            embeddings=[
                embeddings[i]
            ],

            ids=[
                f"{file.filename}_{i}"
            ],

            metadatas=[

                {
                    "source":
                        file.filename
                }

            ]

        )


    # -----------------------------
    # Return success response
    # -----------------------------

    return {

        "message":
            "Document uploaded and processed successfully!",

        "filename":
            file.filename,

        "chunks":
            len(chunks)

    }


# -----------------------------
# Run Flask
# -----------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )