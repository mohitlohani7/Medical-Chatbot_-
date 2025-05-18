from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec, Index
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


# Load and process PDF data
extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Initialize Pinecone and create the index
pc = Pinecone(api_key="pcsk_6Ah8nH_JsPqoT59nZma6BnSXmintSQkLTC5b7nVbcEMhbrF3j5J79sxboS7Nh8uENX5GK8")
index_name = "medicalbot"

# Create Pinecone index if it doesn't already exist
if index_name not in pc.list_indexes():  # Fixed list_indexes() call
    pc.create_index(
        name=index_name,
        dimension=384,  # embedding model dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to the existing Pinecone index
index = Index(index_name)

# Embed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index=index,
    embedding=embeddings
)

print("Embeddings successfully upserted into Pinecone.")
