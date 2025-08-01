# main.py
# Contains the main logic for processing telecom tickets (train, test) and generating visualizations.

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
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("API_VERSION", "2023-05-15"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
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
        api_key=os.getenv('CHROMA_API_KEY'),
        tenant=os.getenv('CHROMA_TENANT'),
        database=os.getenv('CHROMA_DATABASE')
    )
    print("Connected to ChromaDB successfully.")

    telecom_tickets = pd.read_csv('telecom_tickets.csv')
    # Apply preprocess to short description and description
    telecom_tickets[['Short Description', 'Description']] = telecom_tickets.apply(
        lambda row: preprocess_ticket(row['Short Description'], row['Description']),
        axis=1, result_type='expand'
    )
    train, test = telecom_tickets[:int(len(telecom_tickets)*0.8)], telecom_tickets[int(len(telecom_tickets)*0.8):]

    print("Processing training set...")
    labels_and_hours = {}
    testset_labels = []
    # Get closest labels for train set and store hours in labels_and_hours
    for index, row in train.iterrows():
        print(f"Processing row {index + 1}/{len(train)}")
        description = row['Short Description'] + '. ' + row['Description']
        closest_label, closest_definition = get_closest_label_from_chroma_db(description, chroma_client)
        if closest_label not in labels_and_hours:
            labels_and_hours[closest_label] = []
        
        labels_and_hours[closest_label].append(row['Hours'])

        testset_labels.append([row['Case Number'], row['Short Description'], row['Description'], closest_label + ': ' + closest_definition, row['Hours']])

    # For each label, generate a distribution of hours
    # for label, hours in labels_and_hours.items():
    #     if len(hours) > 0:
    #         plt.hist(hours, bins=10, alpha=0.5, label=label)
    #         plt.xlabel('Hours')
    #         plt.ylabel('Frequency')
    #         plt.title(f'Hours Distribution for {label}')
    #         plt.legend()
    #         # Sanitize the label for use as filename
    #         safe_label = re.sub(r'[<>:"/\\|?*]', '_', label)
    #         plt.savefig(f'visualizations/hours_distribution_{safe_label}.png')
    #         plt.clf()  # Clear the figure for the next plot

    # Create a DataFrame for the training set labels
    train_labels_df = pd.DataFrame(testset_labels, columns=['Case Number', 'Short Description', 'Description', 'Label and Definition', 'Hours'])
    train_labels_df.to_csv('train_labels.csv', index=False)
    print("Training set processed and saved to train_labels.csv")

    # Save labels and hours to a CSV file
    labels_and_hours_df = pd.DataFrame(list(labels_and_hours.items()), columns=['Label', 'Hours'])
    labels_and_hours_df.to_csv('labels_and_hours.csv', index=False)
    
    print("Predicting hours for test set...")
    predictions = []
    # Predict number of hours for test set
    for index, row in test.iterrows():
        print(f"Processing row {index + 1}/{len(test)}")
        description = str(row['Short Description']) + ': ' + str(row['Description'])
        closest_label, closest_definition = get_closest_label_from_chroma_db(description, chroma_client)
        if closest_label in labels_and_hours:
            # Predict hours based on training data
            predicted_hours = sum(labels_and_hours[closest_label]) / len(labels_and_hours[closest_label])
            predictions.append([row['Case Number'], row['Short Description'], row['Description'], closest_label + ': ' + closest_definition, predicted_hours, row['Hours']])

    # Create a DataFrame for the predictions
    predictions_df = pd.DataFrame(predictions, columns=['Case Number', 'Short Description', 'Description', 'Predicted Label and Definition', 'Predicted Hours', 'Actual Hours'])
    predictions_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    main()