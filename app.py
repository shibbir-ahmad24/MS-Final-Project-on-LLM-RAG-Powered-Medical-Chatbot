import os
import streamlit as st
import time
from metrics_tracker import MetricsTracker
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
        print(f"Selected Model: {selected_model}")
        print(f"RAG Enabled: {rag_toggle}")

        st.markdown("---")
        
        # Track if upload message already printed
        if "upload_logged" not in st.session_state:
            st.session_state.upload_logged = False
            
        st.markdown("## \U0001F4C2 Upload Data")
        uploaded_files = st.file_uploader("Upload Data", type=["pdf", "txt", "csv"], accept_multiple_files=True, key="file_uploader")
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")
                
            if not st.session_state.upload_logged:
                print(f"{len(uploaded_files)} file(s) uploaded:")
                for uploaded_file in uploaded_files:
                    print(f"    - {uploaded_file.name}")
        
                st.session_state.upload_logged = True  # Prevent re-logging
            
            if st.button("Generate Embeddings", key="generate_embeddings"):
                print(f"Generating embeddings for {len(uploaded_files)} uploaded file(s)...")
                generate_embeddings_from_files(uploaded_files)
                print(f"Embedding generation completed and stored in ChromaDB.")
                st.success("Embeddings generated and stored in ChromaDB.")

        st.markdown("---")
        st.markdown("## \U0001F4A1 Key Features")
        st.markdown("""
        - **BioBERT for Vector Embeddings**
        - **ChromaDB for Vector Storage**
        - **PySpur AI Agent Tools:**
            - Chat Memory Symptom Reasoner
            - Treatment Recommender
            - Symptom Cause Analyzer
            - Notes-Trial Matcher
        """)

    # Initialize tracker
    if "metrics" not in st.session_state:
        st.session_state.metrics = MetricsTracker()

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
            print(f"New user query received: {user_query}")
            start_time = time.time()
            st.session_state.chat_history.append({"user": user_query})

            # Determine tool based on query keywords
            tool_kwargs = {"model": selected_model, "use_rag": rag_toggle}
            routed_correctly = False

            if "symptom" in user_query.lower():
                print("Selected Tool: Symptom Cause Analyzer")
                response = symptom_search_tool(user_query, **tool_kwargs)
                routed_correctly = True
            elif "treatment" in user_query.lower():
                print("Selected Tool: Treatment Recommender")
                response = treatment_tool(user_query, **tool_kwargs)
                routed_correctly = True
            elif "trial" in user_query.lower():
                print("Selected Tool: Notes-Trial Matcher")
                response = trial_matcher_tool(user_query, **tool_kwargs)
                routed_correctly = True
            else:
                print("Selected Tool: Chat Memory Symptom Reasoner")
                response = chat_memory_tool(user_query, **tool_kwargs)
                routed_correctly = True

            st.session_state.chat_history[-1]["response"] = response
            print(f"Response generated successfully for the user query.")

            end_time = time.time()
            response_time = end_time - start_time
            
            # Update metrics tracker
            st.session_state.metrics.record_query(routed_correctly, response_time)
            st.session_state.metrics.print_metrics_summary()
            
            st.rerun()

        # Clear button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            print("Chat history cleared by the user.")
            st.session_state.metrics = MetricsTracker()
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