# Medical QA Chatbot Using PySpur, RAG, and LLMs

## 🩺 Overview

Doctors, researchers, and even patients often need quick answers from complex medical records and clinical trial information. Discharge summaries contain important details about a patient's condition and treatment, while clinical trials offer potential care options — but both can be hard to understand and time-consuming to search through.

This project introduces a smart medical chatbot that uses **PySpur Agent, Retrieval-Augmented Generation (RAG)**, and **Large Language Models (LLMs)** to answer **heart attack/failure** related questions. It works by analyzing real **MIMIC-IV discharge notes** and **clinical trial data**, using **BioBERT** and a **ChromaDB vector store** to find and understand relevant information. The chatbot helps users get accurate, context-aware answers — making medical information easier to access and understand.

## ❗ Problem Statement

Medical discharge summaries and clinical trial descriptions are often packed with complex medical terms and dense formatting, making them hard to understand quickly — even for healthcare professionals. Whether you're trying to extract key information about a patient’s history or find relevant clinical trials, manually digging through these documents takes time and effort.

On top of that, traditional AI chatbots often fall short. They may give answers that sound confident but aren't grounded in real data — a problem known as **hallucination**. In medicine, that kind of inaccuracy isn’t just unhelpful — it can be dangerous.

This project tackles those challenges head-on by building a smarter, safer **medical chatbot**. It combines:

- **BioBERT** for understanding and embedding clinical language,

- LLMs like **LLaMA** and **DeepSeek** for generating natural, helpful answers, and

- **PySpur** and **Retrieval-Augmented Generation (RAG)** to make sure those answers are backed by real discharge notes and clinical trial data.

The result is a reliable AI assistant that understands medical language, pulls information from real-world documents, and responds clearly — helping clinicians, researchers, and patients make better, faster decisions.


## 🎯 Objective

This project focuses on building an AI assistant that helps users understand complex medical information — with a special focus on heart attacks and heart failure — using real discharge summaries and clinical trial data.

**Key objectives include:**

- Use **Retrieval-Augmented Generation (RAG)** to ground answers in real, trusted medical documents
- Embed and search discharge notes and clinical trials using **BioBERT** and **ChromaDB**
- Route queries through PySpur AI agents to four custom tools based on the type of question
- Generate clear, accurate answers using LLMs like **LLaMA-3** and **DeepSeek**
- Compare chatbot performance with and without RAG to evaluate improvements
- Deploy the app on **Hugging Face Spaces** using **Streamlit SDK** for easy access and testing
- Focus specifically on **heart-related** cases — not for diagnosis, but to support understanding and decision-making


## 📊 Data Sources

This chatbot is powered by real-world, clinically relevant datasets:

🏥 Discharge Summaries from the MIMIC-IV Note dataset
- Rich, de-identified patient discharge notes from intensive care units — ideal for training and testing clinical reasoning.
- Link: https://physionet.org/content/mimic-iv-note/2.2/note/

📋 Clinical Trial Data via the ClinicalTrials.gov API
- Up-to-date information on completed clinical trials, including inclusion/exclusion criteria and interventions.
- Link: https://clinicaltrials.gov/data-api/api

## 🧭 Workflow Diagram

![Workflow](https://github.com/shibbir-ahmad24/Medical-QA-Chatbot-Using-PySpur-RAG-and-LLMs/blob/main/Chatbot-workflow.jpg)

## ⚙️ Methodology

This project follows a step-by-step approach to build a smart, clinically relevant chatbot using PySpur, RAG, and LLMs, with a focus on heart attack and heart failure data.

### 📥 1. Data Ingestion & Preprocessing

**Discharge Notes (MIMIC-IV):**
- Extracted over 430,000 patient records and filtered them down to 13,642 with heart-related diagnoses.
- Merged with discharge notes and isolated key sections like History of Present Illness and Hospital Course.
- Resulted in 9,000+ clean notes, chunked into ~26,800 paragraphs for contextual embedding.

**Clinical Trials (ClinicalTrials.gov):**
- Queried and filtered 603 heart attack-related trials, then cleaned inclusion/exclusion criteria.
- Final dataset includes 594 well-structured trials with reliable metadata and clear eligibility information.

### 🔎 2. Embedding & Storage

- **BioBERT Embeddings:**
  - Used BioBERT, a pretrained biomedical transformer, to convert all medical text into vector embeddings.

- **ChromaDB Collections:**

  - 🧾 Discharge Notes: stored with subject_id and hadm_id
  
  - 📋 Clinical Trials: stored with NCT ID and cleaned text blocks

All embeddings are stored in **ChromaDB** for fast, semantic retrieval during live user interaction.

### 🧠 3. Query Understanding & Tool Routing (via PySpur)

When a user submits a question, the system first analyzes the query using keyword-based intent matching (e.g., “symptom”, “treatment”, “trial”).

Once the intent is identified, **PySpur comes into play by managing the registered tool functions**, enabling clean routing and execution.

The system dynamically selects the most relevant tool from four PySpur-registered functions:

- 🔬 Symptom Cause Analyzer – identifies potential causes using trusted web search
- 💊 Treatment Recommender – retrieves relevant discharge notes to suggest treatments
- 🧪 Clinical Trial Matcher – matches patient notes to real-world clinical trials
- 🗂️ Chat Memory Symptom Reasoner – summarizes previously mentioned symptoms

Tool selection is fully automatic — users don’t need to specify anything manually.

### 📡 4. Retrieval-Augmented Generation (RAG)

For tools that use RAG (💊 Treatment Recommender and 🧪 Clinical Trial Matcher):

- The selected tool embeds the user query using BioBERT.
- It queries the ChromaDB vector store to retrieve the most relevant context chunks.
- These retrieved chunks are then combined with the original query to form a grounded, context-rich prompt.
- The final prompt is sent to an LLM (LLaMA or DeepSeek) to generate the answer.

For tools not using RAG (like the 🔬 Symptom Cause Analyzer), alternative mechanisms are used:

- The tool performs a web search via **SerpAPI** on trusted medical sources.
- Retrieved snippets are summarized using the LLM to explain possible causes.

### 🤖 5. Response Generation using LLMs

- The composed prompt is passed to a Large Language Model (LLama 3 or DeepSeek) for response generation.
- The model produces clear, concise, and clinically grounded answers tailored to the user's question and the retrieved context.
- This helps minimize hallucinations and ensures real-world relevance.

### 💬 6. Real-Time Chat Interface (Streamlit)

- The chatbot is built using Streamlit, offering a simple, responsive UI for users to:

  - Ask clinical questions interactively
  - Switch between LLMs and toggle RAG mode
  - View past queries and system responses in chat format
  - Reset the session as needed

🚀 7. Cloud Deployment (Hugging Face Spaces)

- The full application — including embeddings, vector database, model interface, and UI — is deployed to Hugging Face Spaces.
- Users can access the chatbot directly in the browser without login, thanks to:
  - Automatic unzipping and loading of ChromaDB on startup
  - Built-in support for anonymous access and cloud-based execution

## 🧠 How PySpur Works

While I am not using PySpur as a full agent yet, PySpur still plays a key backend role by standardizing and managing my tools. Here's how it works under the hood:

### 1. 🔧 Tool Wrapping with @tool_function
   
Each of my tools (e.g., treatment_tool, trial_matcher_tool) is wrapped using PySpur’s @tool_function decorator.

✅ This allows:

- Automatic creation of structured input/output models
- Consistent function signatures for all tools
- Tools to behave like modular “nodes” — pluggable into larger workflows

### 2. 🗂 Tool Metadata Storage

The decorator also registers important metadata about each tool:
  - Tool name and description
  - Expected input parameters and types
  - Output schema

✅ This metadata can later be used for:
  - Auto-generating interfaces (like UIs)
  - Validating inputs before execution
  - Logging, tracing, or error debugging
  - Seamless orchestration in agent workflows

### 3. 🧩 Enabling Modularity & Future Agents

By standardizing tools now, I am laying the groundwork to:
- Easily compose multi-step workflows
- Let a PySpur Agent dynamically choose tools based on reasoning

## 🛠️ Tech Stack

- **Python:** Main programming language used across all components.
- **BioBERT (via Hugging Face Transformers):** Used to generate domain-specific embeddings for medical text.
- **ChromaDB:** High-performance vector database for storing and retrieving embedded medical data.
- **PySpur:** Provides standardized tool registration and modular execution logic for clean integration of AI tools.
- **Streamlit:** Used to build and deploy the chatbot interface on Hugging Face Spaces.
- **SQL (PostgreSQL):** Initially used for querying structured patient and diagnosis records from the MIMIC-IV database.
- **Jupyter Notebook:** Used during data preprocessing, exploration, and prototyping before system integration.

## 🚀 Live Demo

Try the chatbot live: [🩺 Medical QA Chatbot on Hugging Face Spaces](https://huggingface.co/spaces/shibbir24/Medical_QA_Chatbot)

## 💬 Example Queries
```
- Recommend treatment for a patient with myocardial infarction who received aspirin and oxygen therapy.
- What clinical trials are available for patients with heart failure?
- Explain possible causes of fatigue based on symptom history.
```

## 📦 Installation & Local Run
```
git clone https://github.com/shibbir-ahmad24/Medical-QA-Chatbot-Using-PySpur-RAG-and-LLMs.git
cd Medical_QA_Chatbot
pip install -r requirements.txt
streamlit run app.py
```

## 📸 UI Preview

![image](https://github.com/user-attachments/assets/2c328fdb-d3d0-4194-aa67-abf92c5457de)

![image](https://github.com/user-attachments/assets/184403eb-5b4d-4282-bc00-c55a0748aa82)

![image](https://github.com/user-attachments/assets/eb8b4438-11c5-4c1a-a252-101bff0f14ea)

![image](https://github.com/user-attachments/assets/797e0a3e-2b06-4d41-bb2b-fd24cdb21627)

![image](https://github.com/user-attachments/assets/ae0602fa-993a-42c4-aa52-d10aadd7ae45)





