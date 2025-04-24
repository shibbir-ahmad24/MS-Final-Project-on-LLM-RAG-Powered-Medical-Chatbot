from embedding import get_embedding, discharge_collection
from groq import Client
from serpapi import GoogleSearch
import spacy

client = Client(api_key="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")  # Replace with secure loading in production
nlp = spacy.load("en_core_web_sm")

SYMPTOM_HINTS = [
    "chest pain", "shortness of breath", "fatigue", "dizziness",
    "nausea", "vomiting", "palpitations", "sweating", "jaw pain",
    "arm pain", "back pain", "tightness", "pressure in chest",
    "arrhythmia", "tachycardia", "bradycardia", "angina",
    "edema", "dyspnea", "syncope", "lightheadedness",
    "ejection fraction", "myocardial infarction", "heart failure",
    "cardiomyopathy", "cardiac arrest"
]

# Tool 1: Chat Memory Symptom Reasoner
def chat_memory_tool(memory: str) -> str:
    doc = nlp(memory)
    found_symptoms = set()
    for chunk in doc.noun_chunks:
        for keyword in SYMPTOM_HINTS:
            if keyword in chunk.text.lower():
                found_symptoms.add(chunk.text.strip())

    symptom_context = (
        f"Previously mentioned symptoms include: {', '.join(found_symptoms)}."
        if found_symptoms else "No clear symptoms found in memory."
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a medical assistant summarizing prior symptoms from memory."},
            {"role": "assistant", "content": memory},
            {"role": "user", "content": (
                f"The patient previously reported: {memory}\n\n"
                f"Symptoms extracted: {symptom_context}\n"
                "Please provide a clear, concise, and helpful summary of these symptoms and suggest next steps."
            )}
        ]
    )
    return response.choices[0].message.content

# Tool 2: Treatment Recommender
def treatment_tool(query: str) -> str:
    try:
        query_embedding = get_embedding(query)
        results = discharge_collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents"]
        )
        top_docs = results['documents'][0]
        combined_context = "\n\n".join(top_docs)
        prompt = (
            "You are a helpful medical assistant. Based on the following discharge notes, recommend essential treatment.\n\n"
            f"### Notes:\n{combined_context}\n\n"
            f"### Condition:\n{query}"
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a medically accurate and safety-focused clinical assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Tool 3: Symptom Cause Analyzer
def symptom_search_tool(symptom_description: str) -> str:
    def perform_search(query):
        params = {
            "engine": "google",
            "q": f"{query} possible causes site:mayoclinic.org OR site:webmd.com OR site:nih.gov",
            "api_key": "f61d6383185f187371e695158febad5a0dd4d1c75640bba50e62064503ed1c24"  # replace with your key
        }
        return GoogleSearch(params).get_dict().get("organic_results", [])

    try:
        results = perform_search(symptom_description)
        if not results:
            return "No results."
        snippets = [res['snippet'] for res in results[:3] if 'snippet' in res]
        search_context = "\n\n".join(snippets)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Use the web context below to explain likely causes of the symptom."},
                {"role": "assistant", "content": search_context},
                {"role": "user", "content": f"What could be the cause of: {symptom_description}?"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Search error: {str(e)}"