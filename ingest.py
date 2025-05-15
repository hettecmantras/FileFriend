# This file – Load PDFs → Chunk → Embed → Save to FAISS



from langchain.document_loaders import PyPDFLoader    #PyPDFLoader loads each page of a PDF as a separate Document object (useful for fine-grained retrieval).
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

def ingest_documents(data_dir="data", save_dir="vectorstore"):
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, file))
            pages = loader.load()
            documents.extend(pages)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(save_dir)
    print(f"Saved vectorstore to '{save_dir}'")

if __name__ == "__main__":
    ingest_documents()
