import os
import streamlit as st
import time
from metrics_tracker import MetricsTracker
from tool_handler import (
    chat_memory_tool,
    treatment_tool,
    symptom_search_tool,
    trial_matcher_tool,
)

# Load vector DB from ZIP on startup

import embedding  

# Set page config

st.set_page_config(
    page_title="Medical QA Chatbot",
    layout="wide",
    page_icon="\U0001F3E5",
    initial_sidebar_state="expanded"
)

# Initialize session state

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "metrics" not in st.session_state:
    st.session_state.metrics = MetricsTracker()

# UI layout

with st.container():
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown("## ⚙️ Settings")
        selected_model = st.radio(
            "Select Model",
            ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"],
            key="model_selector",
        )
        rag_toggle = st.toggle("Enable RAG", value=True, key="rag_toggle")
        print(f"Selected Model: {selected_model}")
        print(f"RAG Enabled: {rag_toggle}")
        
        st.markdown("---")

        st.markdown("## 💡 Key Features")
        st.markdown(
            """
        - **BioBERT for Embeddings**
        - **Prebuilt ChromaDB Vector Store**
        - **Tools Available:**
          - Chat Memory Symptom Reasoner
          - Treatment Recommender
          - Symptom Cause Analyzer
          - Clinical Trial Matcher
        """
        )

    with right_col:
        st.markdown(
            """
        <div style='text-align: center; padding-top: 10px;'>
            <h2>❤️ Medical QA Chatbot</h2>
        </div>
        <br><br>
        """,
            unsafe_allow_html=True,
        )

        chat_container = st.container()
        user_query = None

        # Show chat history
        for entry in st.session_state.chat_history:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(entry["user"])
            if "response" in entry:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(entry["response"])

        user_query = st.chat_input("Ask a question...")

        if user_query:
            print(f"New Query: {user_query}")
            start_time = time.time()
            st.session_state.chat_history.append({"user": user_query})

            # Determine tool based on query keywords
            tool_kwargs = {"model": selected_model, "use_rag": rag_toggle}
            routed_correctly = False

            if "symptom" in user_query.lower():
                print("Selected Tool: Symptom Cause Analyzer")
                response = symptom_search_tool(user_query, model=selected_model)  # No RAG for this tool
                routed_correctly = True
            elif "treatment" in user_query.lower():
                print("Selected Tool: Treatment Recommender")
                response = treatment_tool(user_query, **tool_kwargs)  # Uses RAG
                routed_correctly = True
            elif "trial" in user_query.lower():
                print("Selected Tool: Notes-Trial Matcher")
                response = trial_matcher_tool(user_query, **tool_kwargs)  # Uses RAG
                routed_correctly = True
            else:
                print("Selected Tool: Chat Memory Symptom Reasoner")
                response = chat_memory_tool(user_query, model=selected_model)  # No RAG for this tool
                routed_correctly = True

            st.session_state.chat_history[-1]["response"] = response
            print(f"Response generated successfully for the query.")

            end_time = time.time()
            response_time = end_time - start_time
            st.session_state.metrics.record_query(routed_correctly, response_time)
            st.session_state.metrics.print_metrics_summary()
            st.rerun()

        # Clear button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            print("Chat history cleared by the user.")
            st.session_state.metrics = MetricsTracker()
            st.rerun()

# Sidebar styling
st.markdown("""
<style>
    section[data-testid="stSidebar"] > div:first-child {
        border-right: 2px solid #ddd;
        padding-right: 1rem;
    }
</style>
""", unsafe_allow_html=True)