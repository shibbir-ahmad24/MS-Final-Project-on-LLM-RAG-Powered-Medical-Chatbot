# Fix SQLite version for ChromaDB compatibility
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import os
import uuid
import pandas as pd
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
from PyPDF2 import PdfReader
import chromadb
from huggingface_hub import login

# use HuggingFace mirror if internet is slow
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Login to HuggingFace
login(token="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") 

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

# Load BioBERT model and tokenizer
print("Loading BioBERT model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Initialize ChromaDB
chroma_client = chromadb.Client()  # Uses in-memory DB for Streamlit Cloud compatibility
discharge_collection = chroma_client.get_or_create_collection(name="discharge_notes")
trials_collection = chroma_client.get_or_create_collection(name="clinical_trials")

# Embedding function
def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().tolist()

# Store embeddings in ChromaDB
def store_in_chromadb(collection, embeddings, texts, metadata_list):
    ids = [str(uuid.uuid4()) for _ in range(len(texts))]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadata_list)

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
                store_in_chromadb(trials_collection, [embedding], [text], [metadata])
            else:
                store_in_chromadb(discharge_collection, [embedding], [text], [metadata])

            print(f"Embedded and stored: {file.name}")

        except Exception as e:
            print(f"Error processing {file.name}: {e}")
        finally:
            torch.cuda.empty_cache()
