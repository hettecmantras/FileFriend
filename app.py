# app.py

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import tempfile
from io import BytesIO
from dotenv import load_dotenv
import os
import json

load_dotenv()

# ---------------------------- Utility: Relevance Check ----------------------------
def is_question_addressed(query, docs, threshold=0.2):
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    query_keywords = query_words - ENGLISH_STOP_WORDS

    if not query_keywords:
        return False

    all_text = " ".join(doc.page_content.lower() for doc in docs)
    doc_words = set(re.findall(r'\b\w+\b', all_text))

    overlap = query_keywords & doc_words
    overlap_ratio = len(overlap) / len(query_keywords)

    return overlap_ratio >= threshold

# ---------------------------- Prompt Template ----------------------------
prompt_template = """
You are a helpful assistant. Answer the question using the provided context.
If the context is not helpful, say "The answer is not available in the documents."

Context:
{context}

Question:hyhhhhe
{question}
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ---------------------------- LLM ----------------------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ---------------------------- Streamlit UI ----------------------------
st.set_page_config(page_title="📚 PDF Q&A with RAG", layout="wide")
st.title("📚 Ask Questions from your PDF")

uploaded_files = st.file_uploader("📂 Upload your PDFs", type="pdf", accept_multiple_files=True)

query = None
qa_chain = None
retriever = None

if uploaded_files:
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        all_docs.extend(pages)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    custom_vector_db = FAISS.from_documents(docs, embeddings)

    retriever = custom_vector_db.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    st.success("✅ Documents uploaded and indexed successfully!")

    query = st.text_input("Ask a question based on your uploaded PDFs:")

# ---------------------------- Text-to-Speech ----------------------------
def speak_text(text):
    escaped_text = json.dumps(text)
    st.components.v1.html(f"""
    <html>
    <body>
        <script>
            function speak() {{
                var msg = new SpeechSynthesisUtterance({escaped_text});
                msg.pitch = 1;
                msg.rate = 1;
                msg.volume = 1;
                msg.lang = "en-US";
                var voices = window.speechSynthesis.getVoices();
                if (voices.length > 0) {{
                    msg.voice = voices.find(voice => voice.lang === "en-US") || voices[0];
                }}
                window.speechSynthesis.speak(msg);
            }}
            speak();
        </script>
    </body>
    </html>
    """, height=0)

def stop_speaking():
    st.components.v1.html("""
    <script>
        window.speechSynthesis.cancel();
    </script>
    """, height=0)

# ---------------------------- Q&A Execution ----------------------------
if query and qa_chain:
    response = qa_chain(query)
    docs = response["source_documents"]

    if not is_question_addressed(query, docs):
        st.warning(" This topic may not exist in the uploaded documents. Here's a general LLM answer:")
        general_response = llm([HumanMessage(content=query)]).content
        st.markdown("### 🤖 General LLM Answer:")
        st.write(general_response)
        if st.button("🔊 Speak Answer"):
            speak_text(general_response)
        if st.button("🛑 Stop Speaking"):
            stop_speaking()
    else:
        st.markdown("### 📌 Answer:")
        st.write(response["result"])
        if st.button("🔊 Speak Answer"):
            speak_text(response["result"])
        if st.button("🛑 Stop Speaking"):
            stop_speaking()

        st.markdown("### 📄 Source Documents:")
        for i, doc in enumerate(response["source_documents"]):
            st.write(f"**Source {i+1}:** {doc.metadata['source']}")
            st.write(doc.page_content[:300] + "...")
