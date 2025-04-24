import os
import streamlit as st
from embedding import generate_embeddings_from_files
from tool_handler import chat_memory_tool, treatment_tool, symptom_search_tool, trial_matcher_tool

# Set page config
st.set_page_config(
    page_title="Medical QA Chatbot",
    layout="wide",
    page_icon="\U0001F3E5",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat continuity
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create vertical split using container and columns
with st.container():
    left_col, right_col = st.columns([1, 2], gap="large")

    # Left Panel UI - Settings and Upload
    with left_col:
        st.markdown("## ⚙️ Settings")
        selected_model = st.radio("Select Model", ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"], key="model_selector")
        rag_toggle = st.toggle("Enable RAG", value=True, key="rag_toggle")

        st.markdown("---")
        st.markdown("## \U0001F4C2 Upload Data")
        uploaded_files = st.file_uploader("Upload Data", type=["pdf", "txt", "csv"], accept_multiple_files=True, key="file_uploader")
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")
            if st.button("Generate Embeddings", key="generate_embeddings"):
                generate_embeddings_from_files(uploaded_files)
                st.success("Embeddings generated and stored in ChromaDB.")

        st.markdown("---")
        st.markdown("## \U0001F4A1 Key Features")
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
        st.markdown("""
        <div style='text-align: center; padding-top: 10px;'>
            <h2>❤️ Medical QA Chatbot</h2>
        </div>
        <br><br>
        """, unsafe_allow_html=True)

        chat_container = st.container()
        user_query = None

        # Show chat history first
        with chat_container:
            for entry in st.session_state.chat_history:
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(entry["user"])
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(entry["response"])

        # Input query at the end
        user_query = st.chat_input("Ask a question...")

        if user_query:
            st.session_state.chat_history.append({"user": user_query})

            # Determine tool based on query keywords
            tool_kwargs = {"model": selected_model, "use_rag": rag_toggle}

            if "symptom" in user_query.lower():
                response = symptom_search_tool(user_query, **tool_kwargs)
            elif "treatment" in user_query.lower():
                response = treatment_tool(user_query, **tool_kwargs)
            elif "trial" in user_query.lower():
                response = trial_matcher_tool(user_query, **tool_kwargs)
            else:
                response = chat_memory_tool(user_query, **tool_kwargs)

            st.session_state.chat_history[-1]["response"] = response
            st.rerun()

        # Clear button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Apply vertical divider with CSS
st.markdown("""
<style>
    section[data-testid="stSidebar"] > div:first-child {
        border-right: 2px solid #ddd;
        padding-right: 1rem;
    }
</style>
""", unsafe_allow_html=True)