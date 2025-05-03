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

# 🧭 Workflow Diagram

![Workflow](https://github.com/shibbir-ahmad24/MS-Final-Project-on-LLM-RAG-Powered-Medical-Chatbot/blob/main/Medical-chatbot-workflow.jpg)

# ⚙️ Methodology

This project follows a step-by-step approach to build a smart, clinically relevant chatbot using PySpur, RAG, and LLMs, with a focus on heart attack and heart failure data.

## 📥 1. Data Ingestion & Preprocessing

**Discharge Notes (MIMIC-IV):**
- Extracted over 430,000 patient records and filtered them down to 13,642 with heart-related diagnoses.
- Merged with discharge notes and isolated key sections like History of Present Illness and Hospital Course.
- Resulted in 9,000+ clean notes, chunked into ~26,800 paragraphs for contextual embedding.

**Clinical Trials (ClinicalTrials.gov):**
- Queried and filtered 603 heart attack-related trials, then cleaned inclusion/exclusion criteria.
- Final dataset includes 594 well-structured trials with reliable metadata and clear eligibility information.

🔎 2. Embedding & Storage
- **BioBERT Embeddings:**
  - Used BioBERT, a pretrained biomedical transformer, to convert all medical text into vector embeddings.

- **ChromaDB Collections:**

  - 🧾 Discharge Notes: stored with subject_id and hadm_id
  
  - 📋 Clinical Trials: stored with NCT ID and cleaned text blocks

All embeddings are stored in **ChromaDB** for fast, semantic retrieval during live user interaction.

# Tech Stack

- **Python:** Core programming language for development.
- **Hugging Face Transformers:** For integrating large language models (LLMs).
- **PySpur:** For efficient vector database-based retrieval in the RAG pipeline.
- **SQL:** For structured data management and querying.
- **Jupyter Notebook:** For exploration, prototyping, and experimentation.
- **Streamlit:** For deployment and building the user interface (UI).

# Project Deadline 

May 02, 2025
