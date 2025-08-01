# Ticket Effort Estimator

This project uses machine learning to predict the effort required to resolve support tickets based on historical data and similarity matching.

## Features

- **Ticket Categorization**: Automatically categorizes tickets based on descriptions
- **Effort Prediction**: Predicts hours required based on historical data
- **Visualization**: Shows distribution of hours for similar ticket categories
- **Streamlit Web Interface**: User-friendly web application

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

#### For Local Development

Create a `.env` file with your API credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-openai-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-key-here
DEPLOYMENT_NAME=your-embedding-deployment-name
API_VERSION=2023-05-15

# ChromaDB Cloud Configuration
CHROMA_API_KEY=your-chroma-api-key
CHROMA_TENANT=your-chroma-tenant-id
CHROMA_DATABASE=your-chroma-database-name
```

#### For Streamlit Cloud Deployment

1. **Push to GitHub**: Make sure your `.env` file is in `.gitignore` (already configured)
2. **Deploy to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `my_app.py` as your main file
3. **Set Secrets**: In your Streamlit Cloud app settings, go to "Secrets" and add:

```toml
AZURE_OPENAI_ENDPOINT = "https://your-openai-endpoint.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-azure-openai-key-here"
DEPLOYMENT_NAME = "your-embedding-deployment-name"
API_VERSION = "2023-05-15"
CHROMA_API_KEY = "your-chroma-api-key"
CHROMA_TENANT = "your-chroma-tenant-id"
CHROMA_DATABASE = "your-chroma-database-name"
```

## Usage

### Running Locally

```bash
streamlit run my_app.py
```

### Running the Main Script

```bash
python add_to_db.py
```

This will:
1. Read definitions from `definitions.txt`
2. Generate embeddings using Azure OpenAI
3. Store the embeddings and metadata in Azure Cosmos DB

### Querying Embeddings

You can query embeddings programmatically:

```python
from add_to_db import query_embeddings

# Get all embeddings
all_embeddings = query_embeddings()

# Get specific label embedding
bug_embedding = query_embeddings("bug")
```

## Azure Cosmos DB Structure

The script creates:
- **Database**: `keyword_embeddings_db`
- **Container**: `embeddings_container`
- **Partition Key**: `/label`

Each document contains:
- `id`: Unique identifier (same as label)
- `label`: The keyword/category name
- `definition`: Human-readable definition
- `embedding`: Vector embedding from Azure OpenAI

## Benefits of Azure Cosmos DB over ChromaDB

1. **Scalability**: Globally distributed, multi-region replication
2. **Performance**: Low-latency reads/writes with SLA guarantees
3. **Integration**: Native Azure integration with other services
4. **Security**: Enterprise-grade security and compliance
5. **Flexibility**: Multiple APIs (SQL, MongoDB, Cassandra, etc.)
6. **Managed Service**: No infrastructure management required

## Error Handling

The script handles:
- Duplicate documents (upserts existing records)
- Missing environment variables (falls back to hardcoded values)
- Database/container creation (creates if doesn't exist)
