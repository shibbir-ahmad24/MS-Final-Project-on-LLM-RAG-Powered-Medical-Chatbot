# An Intelligent **Medical QA Chatbot** Using PySpur AI Agent, Retrieval-Augmented Generation, and LLMs

# Overview

Healthcare professionals and researchers often need quick and reliable insights from medical records, especially discharge summaries, which contain critical patient information. This project aims to build an **AI-powered medical chatbot** that leverages **PySpur RAG (Retrieval-Augmented Generation) mechanism and Large Language Models (LLMs)** to answer clinical questions based on real-world patient discharge notes from the **MIMIC-IV dataset**. By integrating **advanced NLP techniques**, this chatbot will provide meaningful, **context-aware responses** to help healthcare professionals, researchers, and even patients understand complex medical summaries efficiently.

# Problem Statement

The challenge addressed by this project is the difficulty of accurately understanding and extracting critical patient information from complex medical discharge summaries and clinical trial data. Key issues include:
- Medical discharge summaries are filled with intricate and specialized terminology, making it challenging to extract relevant details effectively.
- Traditional chatbot models often suffer from hallucinations, generating inaccurate or misleading responses when queried about medical data.
- There is a lack of effective, structured retrieval methods to ensure that chatbot responses are grounded in real, clinically relevant data, which is crucial for accurate healthcare decision-making.
  
To overcome these challenges, this project integrates a Retrieval-Augmented Generation (RAG) pipeline to enhance chatbot performance. The system will leverage PySpur’s retrieval mechanism to ensure context-aware and accurate information retrieval, followed by response generation using large language models (LLMs) to minimize hallucinations. The goal is to provide healthcare professionals and researchers with a more accurate, reliable AI assistant that can extract, analyze, and present critical medical data from discharge summaries and clinical trials to support informed decision-making and improve patient care.


# Objective

This project aims to develop a medical chatbot that utilizes a Retrieval-Augmented Generation (RAG) pipeline for enhanced accuracy and context-aware responses. 

The objectives of the project are:
- Develop a medical chatbot that integrates a RAG-based pipeline for retrieving and generating responses from MIMIC-IV discharge summaries and clinical trial data, focusing on heart attack/failure cases.
- Implement a structured retrieval system using BioBERT for embedding medical text and PySpur Vector DB for efficient and relevant information retrieval.
- Evaluate the chatbot’s effectiveness in answering clinical questions related to heart attack/failure discharge summaries, ensuring accurate and reliable information.
- Compare the performance of the chatbot with RAG-based retrieval and generation against the same chatbot using a traditional LLM-based approach to assess improvements in accuracy, relevance, and reduction of hallucinations.
- Deploy the chatbot using Streamlit, providing a user-friendly interface for healthcare professionals and researchers to efficiently interact with the system.

The scope of the chatbot will be confined to understanding and responding to clinical queries about heart attack/failure-related discharge summaries. It will not be designed for real-time diagnosis but will serve as an AI assistant to help medical professionals and researchers navigate and interpret patient records more effectively, thereby supporting clinical decision-making and research.

# Data Source

- discharge note dataset from MIMIC-IV DB: https://physionet.org/content/mimic-iv-note/2.2/note/
- clinical trials from ClinicalTrials.gov API: https://clinicaltrials.gov/data-api/api

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

April 29, 2025
