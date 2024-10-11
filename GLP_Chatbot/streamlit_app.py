import streamlit as st
import openai
import qdrant_client
from qdrant_client.http import models
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Initialize Qdrant client with URL and API key
qdrant_url = os.getenv('QDRANT_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
client = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

# Function to embed text using OpenAI
def embed_text(texts, model="text-embedding-3-large"):
    openai.api_key = os.environ['OPENAI_API_KEY']
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, model=model)
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

# Function to handle user input and generate response
def handle_user_input(query):
    # Embed the query
    query_vector = embed_text([query])[0]
    
    try:
        # Search in Qdrant vector store
        response = client.search(
            collection_name="words-of-the-sequence",
            query_vector=query_vector,
            
        )
        
        # Extract context from search results
        context = " ".join([json.loads(hit.payload["_node_content"]).get("text", "") for hit in response])
        
        # Generate response using OpenAI's GPT-4
        openai.api_key = os.environ['OPENAI_API_KEY']
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\nAnswer:"}
            ],
            max_tokens=1000
        )
        answer = completion.choices[0].message['content'].strip()
        
        return answer
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return ""

# Function to get metadata from Qdrant
def get_metadata(query):
    # Embed the query
    query_vector = embed_text([query])[0]
    
    try:
        # Search in Qdrant vector store
        response = client.search(
            collection_name="words-of-the-sequence",
            query_vector=query_vector,
            limit=5
        )
        
        # Extract metadata from search results
        metadata_set = set()
        for hit in response:
            node_content = json.loads(hit.payload["_node_content"])
            metadata = {
                "page_number": node_content.get("metadata", {}).get("page_number", ""),
                "title": node_content.get("metadata", {}).get("title", "")
            }
            metadata_set.add(json.dumps(metadata))  # Convert dict to string for set uniqueness
        
        # Convert set back to list of dicts
        unique_metadata = [json.loads(meta) for meta in metadata_set]
        
        return unique_metadata
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

# Streamlit app
st.title("OECD GLP Chatbot")

# Initialize session state
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""
if 'response' not in st.session_state:
    st.session_state.response = ""
if 'metadata' not in st.session_state:
    st.session_state.metadata = []

# User input
user_query = st.text_area("Your question:", value=st.session_state.user_query)

# Submit button
if st.button("Submit"):
    if user_query:
        st.session_state.user_query = user_query
        st.session_state.response = handle_user_input(user_query)
        st.session_state.metadata = []  # Clear metadata when new query is submitted
    else:
        st.write("Please enter a query.")

# Display response
if st.session_state.response:
    st.write(st.session_state.response)

# Metadata button
if st.button("Get Metadata"):
    if user_query:
        st.session_state.metadata = get_metadata(user_query)
    else:
        st.write("Please enter a query to get metadata.")

# Display metadata
if st.session_state.metadata:
    st.write(st.session_state.metadata)

# Clear button
if st.button("Clear"):
    st.session_state.user_query = ""
    st.session_state.response = ""
    st.session_state.metadata = []