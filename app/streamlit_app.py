import streamlit as st
import os
from dotenv import load_dotenv
import shutil
import chromadb

from app.rag_pipeline import load_and_process_document, create_vectorstore, load_vectorstore
from app.llm_chain import create_qa_chain

load_dotenv()

st.set_page_config(page_title="Chat with RAG System", layout="wide")
st.title("Intelligent Document Query System")

st.sidebar.title("RAG System Settings")

if st.sidebar.button("Reset Vectorstore"):
    if os.path.exists("db"):
        try:
            chroma_client = chromadb.PersistentClient(path="db")
            chroma_client.reset()
            chroma_client.close()
        except Exception as e:
            st.sidebar.warning(f"Warning closing DB: {e}")
        try:
            shutil.rmtree("db")
            st.sidebar.success("Vectorstore reset successfully.")
        except Exception as e:
            st.sidebar.error(f"Failed to delete DB: {e}")
    else:
        st.sidebar.info("No vectorstore found to reset.")

if not os.path.exists("db"):
    with st.spinner("Processing document and building vectorstore..."):
        docs = load_and_process_document("data/war_and_peace.txt")
        vectordb = create_vectorstore(docs)
else:
    vectordb = load_vectorstore()

qa_chain = create_qa_chain(vectordb)

st.subheader("Ask a question about the document:")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

query = st.chat_input("Type your question...")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Generating answer..."):
        result = qa_chain.invoke(query)
        answer = result.get("result", "No answer found.")

    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

if st.button("Clear Chat History"):
    st.session_state.messages = []
