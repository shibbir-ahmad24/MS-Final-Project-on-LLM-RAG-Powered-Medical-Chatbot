import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import zipfile
import torch
from transformers import AutoModel, AutoTokenizer
import chromadb

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromadb_store")
ZIP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromadb_store.zip")

# Step 1: Unzip the vector store if not already present
if not os.path.exists(os.path.join(DB_DIR, "chroma.sqlite3")):
    print("🔓 Unzipping prebuilt ChromaDB store...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Vector store unzipped and ready.")
else:
    print("Vector store already present. Skipping unzip.")

# Step 3: Connect to persistent ChromaDB
client = chromadb.PersistentClient(path=DB_DIR)
discharge_collection = client.get_or_create_collection("discharge_notes")
trials_collection = client.get_or_create_collection("clinical_trials")

# Step 4: Load BioBERT for embedding
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Step 5: Embedding function
def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().tolist()

# Final check
print(f"📦 ChromaDB Status:")
print(f"  - Discharge Notes Loaded: {discharge_collection.count()}")
print(f"  - Clinical Trials Loaded: {trials_collection.count()}")