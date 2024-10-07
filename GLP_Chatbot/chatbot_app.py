import streamlit as st
from llama_def import create_index, create_query_engine
from llama_def import setup_llm, setup_embed_model, setup_vector_store
from llama_index.core.settings import Settings
from llama_def import create_query_pipeline
from llama_index.core.query_pipeline import InputComponent
from llama_index.core import StorageContext
from dotenv import load_dotenv
from jinja2 import Template
import nltk
import os

nltk.download('punkt_tab')

# Load API keys and URLs from environment variables

load_dotenv()

CO_API_KEY = os.environ['CO_API_KEY'] 
OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] 
QDRANT_URL = os.environ['QDRANT_URL'] 
QDRANT_API_KEY = os.environ['QDRANT_API_KEY'] 
COLLECTION_NAME = "words-of-the-sequence"

# Vector_store set up

setup_llm(
    provider="cohere", 
    model="command-r-plus",
    api_key=CO_API_KEY
    )

setup_embed_model(
    provider="openai", 
    model_name="text-embedding-3-large",
    api_key=OPENAI_API_KEY
    )

vector_store = setup_vector_store(QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#Create Index
index = create_index(
    from_where="vector_store",
    embed_model=Settings.embed_model, 
    vector_store=vector_store 
    )

# Create query engine
query_engine = create_query_engine(index=index, mode="query")

# Create query pipeline
input_component = InputComponent()
chain = [input_component, query_engine]
query_pipeline = create_query_pipeline(chain)

def answer_question(question):
    """
    Answers a question using the pre-built query pipeline.

    Args:
        question (str): The question to be answered.

    Returns:
        str: The formatted answer text.
    """
    response = query_pipeline.run(input=question)
    if response:
        answer_text = response.response
        if answer_text:
            # Use nltk for sentence segmentation
            sentences = nltk.sent_tokenize(answer_text)

            # Create a Jinja2 template
            template = Template("""
                **Key Points:**
                {% for point in sentences %}
                - {{ point }}
                {% endfor %}
            """)

            # Render the template with the sentences
            formatted_response = template.render(sentences=sentences)
            return formatted_response
        else:
            return "The LLM couldn't generate a response for this question."
    else:
        return "Sorry, I couldn't find an answer to your question."
    
st.title("Conversatonal AI Chatbot")
st.subheader("OECD GLP Documents!!!")

# Initialize session state for user_question and response
if 'user_question' not in st.session_state:
    st.session_state.user_question = ""
if 'response' not in st.session_state:
    st.session_state.response = ""
if 'metadata' not in st.session_state:
    st.session_state.metadata = {}

st.session_state.user_question = st.text_area("Ask a question:", value=st.session_state.user_question, height=0)

if st.button("Submit"):
    st.session_state.response = answer_question(st.session_state.user_question)
    st.session_state.metadata = {}  # Clear metadata when a new question is submitted

if st.session_state.response:
    st.write(st.session_state.response)

if st.button("Metadata"):
    response = query_pipeline.run(input=st.session_state.user_question)
    st.session_state.metadata = response.metadata
    st.write(st.session_state.metadata)
        
if st.button("Clear"):
    st.session_state.user_question = ""
    st.session_state.response = ""
    st.session_state.metadata = {}


            