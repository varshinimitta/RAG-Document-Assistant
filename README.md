# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF or TXT documents and ask questions about their content.

## Features

- Upload PDF and TXT documents
- Extract text from documents
- Split documents into smaller chunks
- Generate text embeddings using Sentence Transformers
- Store embeddings in ChromaDB
- Retrieve the top 3 relevant document chunks using semantic search
- Generate answers using Llama 3.2 through Ollama
- Display the source document
- REST API for document upload and question answering
- Simple web interface
- Runs locally without requiring an OpenAI API key

## Technologies Used

- Python
- Flask
- Sentence Transformers
- ChromaDB
- Ollama
- Llama 3.2
- PyPDF
- HTML
- CSS
- JavaScript

## How It Works

```text
PDF / TXT Document
        ↓
   Text Extraction
        ↓
      Chunking
        ↓
   Text Embeddings
        ↓
     ChromaDB
        ↓
   User Question
        ↓
Question Embedding
        ↓
 Semantic Search
        ↓
Top 3 Relevant Chunks
        ↓
    Llama 3.2
        ↓
   Final Answer
        ↓
 Answer + Source
````

## Project Structure

```text
RAG-Question-Answering/
│
├── app.py
├── read_document.py
├── rag_answer.py
├── reset_database.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── templates/
│   └── index.html
│
└── data/
    └── documents/
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd RAG-Question-Answering
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Ollama Setup

Install Ollama and download the Llama 3.2 model:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running before starting the application.

## Run the Application

Start the Flask application:

```bash
python app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

## How to Use

1. Open the application.
2. Upload a PDF or TXT document.
3. The application extracts the document text.
4. The text is divided into smaller chunks.
5. Embeddings are generated using Sentence Transformers.
6. The embeddings are stored in ChromaDB.
7. Enter a question about the uploaded document.
8. The system retrieves the most relevant document chunks.
9. Llama 3.2 generates an answer using the retrieved context.
10. The source document is displayed with the answer.

## API Endpoints

### Upload Document

```text
POST /api/upload
```

The endpoint accepts PDF and TXT files.

### Ask a Question

```text
POST /api/ask
```

Example request:

```json
{
    "question": "What does the company develop?"
}
```

Example response:

```json
{
    "question": "What does the company develop?",
    "answer": "TechNova Solutions develops artificial intelligence and software solutions.",
    "source": "company2.txt"
}
```

## Learning Outcomes

This project demonstrates practical experience with:

* Retrieval-Augmented Generation (RAG)
* Text embeddings
* Vector databases
* Semantic search
* Large Language Models
* Document processing
* REST APIs
* Flask
* Prompt engineering
* Local AI model integration

## Future Improvements

* DOCX document support
* Multiple document management
* Document deletion
* Conversation history
* Improved source citations
* Authentication
* Cloud deployment
* Streaming responses

## Author

Varshini Mitta

B.Tech – Computer Science and Engineering (Data Science)

```
```
