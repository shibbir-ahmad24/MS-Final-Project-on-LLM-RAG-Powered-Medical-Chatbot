import os
import uuid
import pandas as pd
import numpy as np
import faiss
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
from PyPDF2 import PdfReader

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

# Load BioBERT model and tokenizer
print("Loading BioBERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# FAISS Indexes (flat L2 index for simplicity)
discharge_index = faiss.IndexFlatL2(768)
discharge_metadata = []
discharge_texts = []

trial_index = faiss.IndexFlatL2(768)
trial_metadata = []
trial_texts = []

# Embedding function
def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().astype('float32')

# Store embeddings in FAISS
def store_in_faiss(index, embeddings, texts, metadata_list, text_list, meta_list):
    index.add(np.stack(embeddings))
    text_list.extend(texts)
    meta_list.extend(metadata_list)

# PDF text extractor
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# File text extraction
def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        return "\n".join(df.iloc[:, 0].astype(str).tolist())
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""

# Embedding generator from uploaded files
def generate_embeddings_from_files(uploaded_files: List):
    for file in uploaded_files:
        try:
            text = extract_text(file)
            if not text.strip():
                print(f"Empty or unreadable content in {file.name}. Skipping.")
                continue

            embedding = get_embedding(text)
            metadata = {"source": file.name}

            if "trial" in file.name.lower():
                store_in_faiss(trial_index, [embedding], [text], [metadata], trial_texts, trial_metadata)
            else:
                store_in_faiss(discharge_index, [embedding], [text], [metadata], discharge_texts, discharge_metadata)

            print(f"Embedded and stored: {file.name}")

        except Exception as e:
            print(f"Error processing {file.name}: {e}")
        finally:
            torch.cuda.empty_cache()

# helper functions to retrieve top results from FAISS
def query_faiss(index, query_embedding, texts, metadata, top_k=3):
    query_embedding = np.array([query_embedding], dtype='float32')
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(texts):
            results.append({"text": texts[idx], "metadata": metadata[idx]})
    return results
