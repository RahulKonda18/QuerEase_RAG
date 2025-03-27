import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer  # Import the embedding model
import chromadb  # Import ChromaDB client
from llama import generate_response
# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your desired model

# Initialize ChromaDB client and collection using the new configuration
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_docs")  # Replace with your collection name

# Streamlit UI
st.title("QuerEase Swiggy FAQ Chatbot")

# User input
query = st.text_input("Enter your query:")

if query:
    # Encode the query
    query_embedding = embedding_model.encode(query).tolist()

    # Retrieve top 3 most relevant documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    temp = []
    # Display results
    st.subheader("LLama Generated Response:")
    st.write(generate_response(query, " ".join(temp)))

    st.subheader("Top Relevant Documents:")
    for i, metadata in enumerate(results["metadatas"][0]):
        st.write(f"**{i+1}. {metadata['text']}**")
        temp.append(metadata['text'])
   
