# MS Final Project on Medical Chatbot

# Project Title

Developing an **LLM-RAG Powered Medical Chatbot** for Clinical Question Answering

# Overview

Healthcare professionals and researchers often need quick and reliable insights from medical records, especially discharge summaries, which contain critical patient information. This project aims to build an **AI-powered medical chatbot** that leverages **Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs)** to answer clinical questions based on real-world patient discharge notes from the **MIMIC-IV dataset**. By integrating **advanced NLP techniques**, this chatbot will provide meaningful, **context-aware responses** to help healthcare professionals, researchers, and even patients understand complex medical summaries efficiently.

# Problem Statement

Medical discharge summaries are **dense, complex, and filled with medical jargon,** making it challenging for both healthcare providers and patients to extract relevant insights quickly. Traditional chatbots often:
- Struggle with **hallucinations (false or misleading responses).**
- Fail to retrieve **accurate, case-specific knowledge** from structured datasets.
- Lack **domain-specific adaptation** to medical terminology and discharge notes.

This project aims to answer the following key questions:
- **How can we ensure chatbot responses are grounded in real clinical discharge notes?**
- **Does integrating a RAG pipeline improve response accuracy compared to a traditional LLM chatbot?**
- **Do any versions of the chatbot hallucinate or provide false information?**
- **How can we effectively fetch and utilize MIMIC-IV discharge summaries for conditions like Heart Attack & Kidney Disease to improve chatbot reliability?**

# Objective

The primary goal is to develop and evaluate a **medical chatbot** that:

- **Implements a RAG-based pipeline** to fetch relevant discharge notes before generating responses.
- **Supports clinical question answering** for **Heart Attack & Kidney Disease**, ensuring responses are grounded in real clinical data.
- **Minimizes hallucinations and false information** by comparing **RAG vs. non-RAG chatbot responses**.
- **Fetches and processes MIMIC-IV discharge summaries** efficiently.
- **Deploys using Hugging Face Spaces or Streamlit**, integrating **VectorDB** for efficient data retrieval and response generation.

By the end of this project, the chatbot should enhance **medical decision-making** and act as a **reliable AI assistant** in healthcare settings.

# Data Source

- discharge note dataset from MIMIC-IV DB: https://physionet.org/content/mimic-iv-note/2.2/note/
- clinical trials from ClinicalTrials.gov API: https://clinicaltrials.gov/data-api/api

# Working Steps

1. Discharge Notes Collection from MIMIC-IV Database
2. Clinical Trials Retrieval from ClinicalTrials.gov API
3. Data Preprocessing (Clean, Segment, and Categorize)
4. Handling User Queries for Clinical Question Answering
   - to design and optimize the system for answering Heart Attack and Kidney Disease-related queries
5. Vectorization using BioBERT Embedding Model
6. Vectorstore Setup for Efficient Retrieval
7. Query Handling with RAG
   - to implement RAG pipeline to retrieve relevant information based on user query
8. Response Generation using LLM
   - to generate responses based on the retrieved data, ensuring clarity, accuracy, and relevance in answering the user’s query.
10. Evaluation by Comparing RAG and Non-RAG Based LLM Models
11. User Interface Design and Deployment
    - to design and deploy a clean, intuitive interface using Streamlit, enabling healthcare professionals to easily interact with the chatbot.

# RAG Pipeline

![RAG Pipeline](https://github.com/shibbir-ahmad24/MS-Final-Project-on-LLM-RAG-Powered-Medical-Chatbot/blob/main/RAG-Pipeline2.jpg)

# Tech Stack

- Python (Core programming language)
- Hugging Face Transformers (LLM integration)
- Faiss / ChromaDB (Vector database for retrieval)
- SQL (Structured data management)
- Jupyter Notebook (Exploration & prototyping)
- Streamlit (Deployment & UI)
- Docker (Containerization for deployment)

# Project Deadline 

April 30, 2025
