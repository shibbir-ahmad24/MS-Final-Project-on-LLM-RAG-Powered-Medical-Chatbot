import os
import pandas as pd
import chromadb
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
from PyPDF2 import PdfReader

# Load BioBERT model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chromadb_store")
discharge_collection = chroma_client.get_or_create_collection(name="discharge_notes")

# Embedding function
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist()

# Text extractor for PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Text extraction from file
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

# Generate embeddings from uploaded files and store to ChromaDB
def generate_embeddings_from_files(uploaded_files: List):
    for file in uploaded_files:
        try:
            text = extract_text(file)
            if not text.strip():
                print(f"⚠️ Empty or unreadable content in {file.name}. Skipping.")
                continue

            embedding = get_embedding(text)
            discharge_collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[{"source": file.name}]
            )
            print(f"✅ Successfully embedded and stored: {file.name}")

        except Exception as e:
            print(f"❌ Error processing {file.name}: {str(e)}")