# Medical QA Chatbot Using PySpur AI Agent, RAG, and LLMs

# 🩺 Overview

Doctors, researchers, and even patients often need quick answers from complex medical records and clinical trial information. Discharge summaries contain important details about a patient's condition and treatment, while clinical trials offer potential care options — but both can be hard to understand and time-consuming to search through.

This project introduces a smart medical chatbot that uses **PySpur AI agent, Retrieval-Augmented Generation (RAG)**, and **Large Language Models (LLMs)** to answer **heart attack/failure** related questions. It works by analyzing real **MIMIC-IV discharge notes** and **clinical trial data**, using **BioBERT** and a **ChromaDB vector store** to find and understand relevant information. The chatbot helps users get accurate, context-aware answers — making medical information easier to access and understand.

# ❗ Problem Statement

Medical discharge summaries and clinical trial descriptions are often packed with complex medical terms and dense formatting, making them hard to understand quickly — even for healthcare professionals. Whether you're trying to extract key information about a patient’s history or find relevant clinical trials, manually digging through these documents takes time and effort.

On top of that, traditional AI chatbots often fall short. They may give answers that sound confident but aren't grounded in real data — a problem known as **hallucination**. In medicine, that kind of inaccuracy isn’t just unhelpful — it can be dangerous.

This project tackles those challenges head-on by building a smarter, safer **medical chatbot**. It combines:

- **BioBERT** for understanding and embedding clinical language,

- LLMs like **LLaMA** and **DeepSeek** for generating natural, helpful answers, and

- **PySpur** and **Retrieval-Augmented Generation (RAG)** to make sure those answers are backed by real discharge notes and clinical trial data.

The result is a reliable AI assistant that understands medical language, pulls information from real-world documents, and responds clearly — helping clinicians, researchers, and patients make better, faster decisions.


# 🎯 Objective

This project focuses on building an AI assistant that helps users understand complex medical information — with a special focus on heart attacks and heart failure — using real discharge summaries and clinical trial data.

**Key objectives include:**

- Use **Retrieval-Augmented Generation (RAG)** to ground answers in real, trusted medical documents
- Embed and search discharge notes and clinical trials using **BioBERT** and **ChromaDB**
- Route queries through PySpur AI agents to four custom tools based on the type of question
- Generate clear, accurate answers using LLMs like **LLaMA-3** and **DeepSeek**
- Compare chatbot performance with and without RAG to evaluate improvements
- Deploy the app on **Hugging Face Spaces** using **Streamlit SDK** for easy access and testing
- Focus specifically on **heart-related** cases — not for diagnosis, but to support understanding and decision-making


# 📊 Data Sources

This chatbot is powered by real-world, clinically relevant datasets:

🏥 Discharge Summaries from the MIMIC-IV Note dataset
- Rich, de-identified patient discharge notes from intensive care units — ideal for training and testing clinical reasoning.
- Link: https://physionet.org/content/mimic-iv-note/2.2/note/

📋 Clinical Trial Data via the ClinicalTrials.gov API
- Up-to-date information on completed clinical trials, including inclusion/exclusion criteria and interventions.
- Link: https://clinicaltrials.gov/data-api/api

# Workflow Diagram

![Workflow](https://github.com/shibbir-ahmad24/MS-Final-Project-on-LLM-RAG-Powered-Medical-Chatbot/blob/main/Medical-chatbot-workflow.jpg)

# Methodology Overview

This project follows a structured approach to developing a PySpur-powered RAG chatbot for clinical question answering. 

The methodology involves:
- **Data Ingestion & Preprocessing:** Extracting discharge notes from MIMIC-IV dataset and clinical trial data from ClinicalTrials.gov API, followed by cleaning, segmenting, and categorizing the text for better structuring.
- **Embedding & Storage:** Using BioBERT for medical text vectorization and storing embeddings efficiently in PySpur Vector DB to enable fast and relevant retrieval.
- **Query Processing & Retrieval:** When a user submits a medical query, PySpur’s retrieval mechanism fetches the most relevant top-K text chunks from the Vector DB, ensuring context-aware information retrieval.
- **Response Generation with LLM:** The retrieved information is passed to an LLM, which generates accurate, relevant, and contextually grounded responses while minimizing hallucinations.
- **Evaluation & Debugging:** PySpur’s built-in execution tracing and evaluation tools will be used to compare RAG-based responses vs. standard LLM responses, refining the chatbot’s accuracy.
- User Interface & Deployment: The chatbot will be deployed using Streamlit, providing healthcare professionals and researchers with an intuitive interface for seamless interaction.

By integrating PySpur’s RAG pipeline, this methodology ensures that the chatbot not only retrieves the most relevant medical information but also generates precise and reliable responses tailored to Heart Attack/Failure related queries.

# Tech Stack

- **Python:** Core programming language for development.
- **Hugging Face Transformers:** For integrating large language models (LLMs).
- **PySpur:** For efficient vector database-based retrieval in the RAG pipeline.
- **SQL:** For structured data management and querying.
- **Jupyter Notebook:** For exploration, prototyping, and experimentation.
- **Streamlit:** For deployment and building the user interface (UI).

# Project Deadline 

May 02, 2025
