import os
import time
import PyPDF2
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import GROQ_API
from prompt import get_prompt_template
from embeddings import generate_embeddings

# New imports for local Mistral chatbot
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from db_utils import verify_credentials, get_student_context
from prompt_mistral import get_mistral_prompt

# -----------------------------------------------------------------------------
# 1) Load LLMs
# -----------------------------------------------------------------------------

# Public RAG LLM
llm = ChatGroq(groq_api_key=GROQ_API, model_name="llama-3.3-70b-versatile")

@st.cache_resource
def load_local_model():
    model_dir = os.path.join("models", "final_quantized")

    # Tokenizer from HF Hub
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        use_fast=True
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Local quantized model weights
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    model.eval()
    return tokenizer, model

tokenizer_local, model_local = load_local_model()

# -----------------------------------------------------------------------------
# 2) Sidebar: Student login / logout
# -----------------------------------------------------------------------------
st.sidebar.title("Student Login")

if "user" not in st.session_state:
    sid = st.sidebar.text_input("Student ID")
    pwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if verify_credentials(sid, pwd):
            st.session_state.user = sid
            st.sidebar.success(f"Logged in as {sid}")
        else:
            st.sidebar.error("Invalid ID or password")
else:
    if st.sidebar.button("Logout"):
        del st.session_state.user
        st.sidebar.info("Logged out")

# -----------------------------------------------------------------------------
# 3) Main: route between public RAG vs. personal schedule chatbot
# -----------------------------------------------------------------------------
if "user" in st.session_state:
    # ----- Personal schedule chatbot -----
    st.title(f"Hello {st.session_state.user}, ask me any questions you may have")
    query = st.text_input("Your question:")

    if st.button("Send") and query:
        # Fetch student details & courses
        student_name, advisor, courses, total_credits = get_student_context(st.session_state.user)

        # Build prompt
        prompt = get_mistral_prompt(
            student_id=st.session_state.user,
            student_name=student_name,
            advisor_name=advisor,
            courses=courses,
            total_credits=total_credits,
            query=query
        )

        # Inference
        inputs = tokenizer_local(prompt, return_tensors="pt", padding=True).to(model_local.device)
        outputs = model_local.generate(
            **inputs,
            max_new_tokens=500,
            pad_token_id=tokenizer_local.eos_token_id
        )
        raw = tokenizer_local.decode(outputs[0], skip_special_tokens=True)

        # 1) Strip out the prompt echo if present
        if raw.startswith(prompt):
            candidate = raw[len(prompt):].strip()
        else:
            candidate = raw.strip()

        # 2) Remove any leading "Answer:" prefix
        if candidate.lower().startswith("answer:"):
            candidate = candidate[len("answer:"):].strip()

        # 3) Trim off anything from the next "Question:" onward
        if "Question:" in candidate:
            candidate = candidate.split("Question:")[0].strip()

        # Final cleaned answer
        answer = candidate

        st.subheader("Answer:")
        st.write(answer)

else:
    # ----- Public RAG-based chatbot -----
    st.title("ROWAN Univeristy Chatbot Prototype")

    PDF_PATH = "document.pdf"
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "raw_docs" not in st.session_state:
        st.session_state.raw_docs = None

    if st.button("Generate Embeddings"):
        try:
            with open(PDF_PATH, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                PDF_TEXT = "".join(page.extract_text() for page in reader.pages)
            st.session_state.vectorstore, st.session_state.raw_docs = generate_embeddings(PDF_TEXT)
            st.success("Embeddings ready and stored in the vector database!")
        except FileNotFoundError:
            st.error(f"File '{PDF_PATH}' not found.")

    def hybrid_retrieval(query, vectorstore, raw_docs, top_k=4):
        vector_results = vectorstore.similarity_search(query, k=top_k)
        keyword_results = [doc for doc in raw_docs if query.lower() in doc.page_content.lower()][:top_k]
        combined = {doc.page_content: doc for doc in (vector_results + keyword_results)}
        return list(combined.values())[:top_k]

    with st.form("public_form"):
        user_query = st.text_input("Enter your question about Rowan University:")
        submitted   = st.form_submit_button("Send")

    if submitted:
        if st.session_state.vectorstore and st.session_state.raw_docs:
            document_chain = create_stuff_documents_chain(llm, get_prompt_template())
            with st.spinner("Processing your query..."):
                start = time.process_time()
                docs  = hybrid_retrieval(user_query, st.session_state.vectorstore, st.session_state.raw_docs)
                resp  = document_chain.invoke({'context': docs, 'input': user_query})
                elapsed = time.process_time() - start

            st.subheader("Answer:")
            st.write(resp)
            st.write(f"⏱ {elapsed:.2f}s")

            with st.expander("Relevant Chunks"):
                for i, doc in enumerate(docs, 1):
                    st.write(f"Chunk {i}: {doc.page_content}")
                    st.write("---")
            st.info("Hybrid: 50% vector + 50% keyword")
        else:
            st.error("Click ‘Generate Embeddings’ to load the document.")
