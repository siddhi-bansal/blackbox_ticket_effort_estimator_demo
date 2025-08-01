import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()

def embed_definitions(definitions):
    """
    Embeds the definitions using Azure OpenAI's embedding model.
    
    Args:
        definitions (dict): A dictionary where keys are labels and values are their definitions.
    
    Returns:
        dict: A dictionary with labels as keys and their embeddings as values.
    """

    # Connect to Azure OpenAI
    openai_client = AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY', 'your-api-key-here'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2023-05-15'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', 'https://oai-snowtee-poc.openai.azure.com/')
    )

    embedded_definitions = {}

    for label, definition in definitions.items():
        response = openai_client.embeddings.create(
            input=f"{label}: {definition}",
            model="text-embedding-ada-002"
        )
        embedded_definitions[label] = response.data[0].embedding

    return embedded_definitions

def main():
    # Define label and definitions, read dictionary from definitions.txt
    definitions = {}

    with open('definitions.txt', 'r') as file:
        definitions = eval(file.read())

    # Embed all definitions with OpenAI
    embedded_definitions = embed_definitions(definitions)
    print("Definitions have been embedded")
        
    # # Connect to ChromaDB
    chroma_client = chromadb.CloudClient(
        api_key='ck-FwM3PFm1bFSZaf6hLKkJ8dZcj5S4amDnxc8Kz4gEV8uN',
        tenant='32e0f1cd-477f-4b7e-8dc4-d6c8a0b36d8c',
        database='snowtee-poc'
    )
    print("Connected to ChromaDB successfully.")

    # Add keywords and their embeddings to the database
    collection = chroma_client.create_collection(name="keyword_embeddings")
    for label, embedding in embedded_definitions.items():
        collection.add(
            documents=[label],
            embeddings=[embedding],
            metadatas=[{"definition": definitions[label]}],
            ids=[label]
        )
    print("Keywords and their embeddings have been added to the database.")

if __name__ == "__main__":
    main()