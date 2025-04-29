"""
Main application for Document Q&A with GROQ and Google Embeddings
"""
import time
import PyPDF2
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import GROQ_API
from prompt import get_prompt_template
from embeddings import generate_embeddings

st.title("SynergenX Chatbot Prototype")

PDF_PATH = "document.pdf"
llm = ChatGroq(groq_api_key=GROQ_API, model_name="llama-3.3-70b-versatile")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = None

if st.button("Generate Embeddings"):
    try:
        with open(PDF_PATH, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            PDF_TEXT = ""
            for page in reader.pages:
                PDF_TEXT += page.extract_text()

        st.session_state.vectorstore, st.session_state.raw_docs = generate_embeddings(PDF_TEXT)
        st.write("Embeddings are ready and stored in the vector database!")
    except FileNotFoundError:
        st.error(f"File '{PDF_PATH}' not found in the directory.")

# Hybrid retrieval function
def hybrid_retrieval(query, vectorstore, raw_docs, top_k=4):
    """Combine vector similarity and keyword matching to retrieve relevant documents."""
    # Cosine similarity search
    vector_results = vectorstore.similarity_search(query, k=top_k)

    # Keyword match search (case-insensitive)
    keyword_results = [doc for doc in raw_docs if query.lower() in doc.page_content.lower()]
    keyword_results = keyword_results[:top_k]

    # Combine and deduplicate results
    combined = {doc.page_content: doc for doc in (vector_results + keyword_results)}
    return list(combined.values())[:top_k]

with st.form("my_form"):
    user_query = st.text_input("Enter your question based on the documentation:")
    submitted = st.form_submit_button("Send")

if submitted:
    if st.session_state.vectorstore and st.session_state.raw_docs:
        document_chain = create_stuff_documents_chain(llm, get_prompt_template())

        with st.spinner("Retrieving and processing your query..."):
            start_time = time.process_time()
            combined_docs = hybrid_retrieval(user_query, st.session_state.vectorstore, st.session_state.raw_docs)
            response = document_chain.invoke({'context': combined_docs, 'input': user_query})           
            end_time = time.process_time() - start_time

            st.subheader("Answer:")
            st.write(response)
            st.write(f"Response time: {end_time:.2f} seconds")

            with st.expander("Relevant Document Chunks"):
                for i, doc in enumerate(combined_docs):
                    st.write(f"Chunk {i + 1}: {doc.page_content}")
                    st.write("---")       
        st.info("Hybrid search: 50% vector similarity + 50% keyword matching.")
    else:
        st.error("Please click 'Generate Embeddings' to load the document.")
