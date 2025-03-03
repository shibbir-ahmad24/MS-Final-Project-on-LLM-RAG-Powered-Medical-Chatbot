# CSIT-697-MS-Project-Medical-Chatbot

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


# System Workflow

# Tech Stack

- Python (Core programming language)
- Hugging Face Transformers (LLM integration)
- Faiss / ChromaDB (Vector database for retrieval)
- SQL (Structured data management)
- Jupyter Notebook (Exploration & prototyping)
- Streamlit / Hugging Face Spaces (Deployment & UI)

# Advisor

Dr. Hao Liu

Assistant Professor, School of Computing,

Montclair State University, New Jersey.

# Project Deadline 

April 30, 2025
