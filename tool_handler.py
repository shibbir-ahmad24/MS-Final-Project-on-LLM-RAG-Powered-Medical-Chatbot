import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'pyspur/backend/')))

from embedding import discharge_collection, trials_collection, get_embedding
from groq import Client
from serpapi import GoogleSearch
import spacy
from pyspur.backend.pyspur.nodes.decorator import tool_function

# Initialize LLM client and spaCy
client = Client(api_key="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")
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

@tool_function(name="chat_memory_tool")
def chat_memory_tool(memory: str, model: str = "llama-3.3-70b-versatile") -> str:
    doc = nlp(memory)
    found_symptoms = set(
        keyword for chunk in doc.noun_chunks for keyword in SYMPTOM_HINTS if keyword in chunk.text.lower()
    )
    symptom_context = (
        f"Previously mentioned symptoms include: {', '.join(found_symptoms)}."
        if found_symptoms else "No clear symptoms found in memory."
    )
    response = client.chat.completions.create(
        model=model,
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

@tool_function(name="treatment_tool")
def treatment_tool(query: str, model: str = "llama-3.3-70b-versatile", use_rag: bool = True) -> str:
    try:
        query_embedding = get_embedding(query)
        if use_rag:
            results = discharge_collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents"]
            )
            top_docs = results['documents'][0] if results and results['documents'] else []
            top_docs = [doc[:1500] for doc in top_docs]
            combined_context = "\n\n".join(top_docs)
            prompt = (
                "You are a helpful medical assistant. Based on the following discharge notes, "
                "recommend essential treatment.\n\n"
                f"### Notes:\n{combined_context}\n\n### Condition:\n{query}"
            )
        else:
            prompt = f"Patient condition: {query}. What treatment is recommended?"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medically accurate and safety-focused clinical assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

@tool_function(name="symptom_search_tool")
def symptom_search_tool(symptom_description: str, model: str = "llama-3.3-70b-versatile") -> str:
    def perform_search(query):
        params = {
            "engine": "google",
            "q": f"{query} possible causes site:mayoclinic.org OR site:webmd.com OR site:nih.gov",
            "api_key": "f61d6383185f187371e695158febad5a0dd4d1c75640bba50e62064503ed1c24"
        }
        return GoogleSearch(params).get_dict().get("organic_results", [])

    try:
        results = perform_search(symptom_description)
        if not results:
            return "No reliable medical source found."

        sources = []
        snippets_with_citations = []
        for res in results[:3]:
            if 'snippet' in res and 'link' in res:
                source_url = res['link']
                domain = source_url.split("//")[-1].split("/")[0].replace("www.", "")
                snippets_with_citations.append(f"{res['snippet']} (Source: {domain})")
                sources.append(source_url)

        search_context = "\n\n".join(snippets_with_citations)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medical assistant using trusted web sources to explain symptom causes."},
                {"role": "assistant", "content": search_context},
                {"role": "user", "content": f"What could be the cause of: {symptom_description}?"}
            ]
        )

        bulleted_sources = "\n".join(f"- {url}" for url in sources)
        return response.choices[0].message.content + "\n\n**Sources:**\n" + bulleted_sources

    except Exception as e:
        return f"Search error: {str(e)}"

@tool_function(name="trial_matcher_tool")
def trial_matcher_tool(discharge_note: str, model: str = "llama-3.3-70b-versatile", use_rag: bool = True) -> str:
    try:
        query_embedding = get_embedding(discharge_note)
        results = trials_collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas"]
        )
        if not results.get('documents') or not results['documents'][0]:
            return "No matching clinical trials were found for the provided note."

        summaries = []
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            nct_id = meta.get("NCT ID") or "Unknown ID"
            truncated_doc = doc.strip()[:1500]
            if use_rag:
                summary_prompt = (
                    f"You are a clinical assistant reviewing a matched clinical trial.\n"
                    f"Summarize the trial using **bullet points only** for the following fields:\n"
                    f"- NCT ID\n- Study Title\n- Conditions\n- Inclusion Criteria\n- Exclusion Criteria\n\n"
                    f"Use bullets under each field. Maintain a clean format. Respond only with the summary.\n\n"
                    f"Trial Description:\nNCT ID: {nct_id}\n{truncated_doc}"
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a medically precise clinical research assistant."},
                        {"role": "user", "content": summary_prompt}
                    ]
                )
                summaries.append(f"### Trial {i+1}:\n{response.choices[0].message.content}")
            else:
                summaries.append(f"### Trial {i+1}:\nNCT ID: {nct_id}\n\n{truncated_doc}")

        return "\n\n---\n\n".join(summaries)

    except Exception as e:
        return f"Error during trial matching: {str(e)}"

# Tool routing via keyword logic
TOOL_ROUTER = {
    "symptom": ("symptom_search_tool", False),
    "treatment": ("treatment_tool", True),
    "trial": ("trial_matcher_tool", True)
}

TOOL_FUNCTIONS = {
    "chat_memory_tool": chat_memory_tool,
    "treatment_tool": treatment_tool,
    "symptom_search_tool": symptom_search_tool,
    "trial_matcher_tool": trial_matcher_tool
}

def run_tool(query: str, model: str, use_rag: bool) -> str:
    for keyword, (tool_name, supports_rag) in TOOL_ROUTER.items():
        if keyword in query.lower():
            print(f"Tool selected by PySpur: {tool_name}")
            tool_func = TOOL_FUNCTIONS[tool_name]
            if supports_rag:
                return tool_func(query, model=model, use_rag=use_rag)
            else:
                return tool_func(query, model=model)

    print("Tool selected by PySpur: chat_memory_tool")
    return chat_memory_tool(query, model=model)