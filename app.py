import os
import streamlit as st
from embedding import generate_embeddings_from_files
from tool_handler import chat_memory_tool, treatment_tool, symptom_search_tool

# Set page config
st.set_page_config(page_title="Clinical QA Chatbot", layout="wide")

# Define the main layout with two columns
left_col, right_col = st.columns([1, 2])

# Left Panel UI - Settings and Upload
with left_col:
    st.markdown("## ⚙️ Settings")
    selected_model = st.radio("Select Model", ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"])
    rag_toggle = st.toggle("Enable RAG (Retrieval-Augmented Generation)", value=True)

    uploaded_files = st.file_uploader("Upload Discharge Notes", type=["pdf", "txt", "csv"], accept_multiple_files=True)
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")
        if st.button("Generate Embeddings"):
            generate_embeddings_from_files(uploaded_files)
            st.success("Embeddings generated and stored in ChromaDB.")

    st.markdown("### Key Features")
    st.markdown("""
    - **BioBERT for Embeddings**
    - **ChromaDB for Vector Storage**
    - **PySpur AI Agent Tools:**
        - Chat Memory Symptom Reasoner
        - Treatment Recommender
        - Symptom Cause Analyzer
        - Notes-Trial Matcher
    """)

# Right Panel UI - Chatbot
with right_col:
    st.title("Clinical QA Chatbot")
    user_query = st.text_area("Ask a clinical question:")
    run_button = st.button("Submit")
    st.markdown("---")

    if run_button and user_query:
        if "symptom" in user_query.lower():
            response = symptom_search_tool(user_query)
        elif "treatment" in user_query.lower():
            response = treatment_tool(user_query)
        else:
            response = chat_memory_tool(user_query)

        st.markdown("### 💬 Response")
        st.write(response)
    else:
        st.info("Enter a query and press 'Submit' to get a response.")