import chromadb
from openai import AzureOpenAI
import pandas as pd
import matplotlib.pyplot as plt
import re
import spacy
from spacy.util import filter_spans
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load spaCy model globally
_NLP = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])

def get_closest_label_from_chroma_db(description, chroma_client):
    """
    Retrieves the closest label from ChromaDB based on the provided description.
    
    Args:
        description (str): The description to match against labels in the database.
        chroma_client (ChromaClient): An instance of the ChromaDB client.
    
    Returns:
        str: The label that is closest to the provided description.
    """
    # Connect to the collection where embeddings are stored
    collection = chroma_client.get_collection(name="keyword_embeddings")
    
    # Embed the description using Azure OpenAI's embedding model
    openai_client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY', 'your-api-key-here'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2023-05-15'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', 'https://oai-snowtee-poc.openai.azure.com/')
    )
    response = openai_client.embeddings.create(
        input=description,
        model="text-embedding-ada-002"
    )
    description_embedding = response.data[0].embedding
    
    # Query the collection for the closest match
    results = collection.query(
        query_embeddings=[description_embedding],
        n_results=1
    )
    
    # Return the label and definition of the closest match
    closest_label = results['documents'][0][0]
    closest_definition = results['metadatas'][0][0]['definition']
    return closest_label, closest_definition

def preprocess_ticket(short_desc: str, desc: str) -> tuple[str, str]:
    """
    Pass raw short description and description.
    Returns: (clean_short_desc, clean_desc)
    """
    # --- Clean short description ---
    external_tags = ("[EXTERNAL] RE:", "[EXTERNAL]")
    if not isinstance(short_desc, str):
        short_desc = ""
    for tag in external_tags:
        short_desc = short_desc.replace(tag, "")
    clean_short = short_desc.replace("||", "").strip()
 
    # --- Clean description ---
    if not isinstance(desc, str):
        desc = ""
 
    boilerplate_patterns = [
        r"\breceived from: ?", r"\bemail received from: ?", r"\bsubject: ?"
    ]
    for pat in boilerplate_patterns:
        desc = re.sub(pat, "", desc, flags=re.IGNORECASE)
 
    # truncate after "Upcoming Time off"
    cut = re.search(r"upcoming time off", desc, flags=re.IGNORECASE)
    if cut:
        desc = desc[:cut.start()]
 
    # remove PERSON names via spaCy
    doc = _NLP(desc)
    for span in reversed(filter_spans([e for e in doc.ents if e.label_ == "PERSON"])):
        desc = desc[:span.start_char] + " " + desc[span.end_char:]
 
    # regex scrubbing
    patterns = [
        r"[^<\n]*<[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}>",  # email in <>
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",        # email
        r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",     # phone
        r"\b\d{5}(?:-\d{4})?\b",                                     # zip
        r"\b\d{1,2}:\d{2}(?:\s?[ap]m)?\b",                           # time
        r"\b\d{4}-\d{2}-\d{2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\b",    # date
        r"\d{1,5}\s+\w+(?:\s+\w+)*\s+\b(?:street|st|road|rd|avenue|ave|boulevard|blvd|"
        r"lane|ln|drive|dr|court|ct|parkway|pkwy|suite|apt|unit|floor|fl)\b",
        r"\[cid:[^\]]+\]",       # images
        r"[-_]{3,}",             # separators
    ]
    for pat in patterns:
        desc = re.sub(pat, "", desc, flags=re.IGNORECASE)
 
    sign_offs = [
        r"\bthank(s| you)[\s,!.\n]*", r"\bbest[\s,!.\n]*", r"\bregards[\s,!.\n]*",
        r"\bsincerely[\s,!.\n]*", r"\bcheers[\s,!.\n]*", r"\byours truly[\s,!.\n]*",
        r"\bkind regards[\s,!.\n]*", r"\bgood morning[\s,!.\n]*", r"\bhello[\s,!.\n]*",
    ]
    for s in sign_offs:
        desc = re.sub(s, "", desc, flags=re.IGNORECASE)
 
    clean_desc = desc.replace("||", "")
    clean_desc = re.sub(r"\s+", " ", clean_desc).strip()
 
    return clean_short, clean_desc

def main():
    chroma_client = chromadb.CloudClient(
        api_key='ck-FwM3PFm1bFSZaf6hLKkJ8dZcj5S4amDnxc8Kz4gEV8uN',
        tenant='32e0f1cd-477f-4b7e-8dc4-d6c8a0b36d8c',
        database='snowtee-poc'
    )
    print("Connected to ChromaDB successfully.")

    telecom_tickets = pd.read_csv('telecom_tickets.csv')[:113]
    # Apply preprocess to short description and description
    telecom_tickets[['Short Description', 'Description']] = telecom_tickets.apply(
        lambda row: preprocess_ticket(row['Short Description'], row['Description']),
        axis=1, result_type='expand'
    )
    data = telecom_tickets
    num_correct = 0
    num_total = 0

    for index, row in data.iterrows():
        print(f"Processing row {index + 1}/{len(data)}")
        description = str(row['Short Description']) + ': ' + str(row['Description'])
        correct_label = f"Telecom - {row['Tagged Issue']}"
        closest_label, closest_definition = get_closest_label_from_chroma_db(description, chroma_client)
        if closest_label.lower() == correct_label.lower():
            num_correct += 1
        else:
            print(f"Row {index + 1}: Expected '{correct_label}', got '{closest_label}'")
        num_total += 1

    accuracy = num_correct / num_total if num_total > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()