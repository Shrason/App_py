from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st

embedding_model = GoogleGenerativeAIEmbeddings(google_api_key='AIzaSyDPuLLK_0j5B1yDRvUcZQFcVibSx__yZiU',model='models/embedding-001')
db_connection = Chroma(persist_directory='rag_db_ copy', embedding_function=embedding_model)
retriever = db_connection.as_retriever(search_kwargs={"k":5})

st.title('RAG App')

user_input = st.text_input('Enter your query here')

if st.button('Submit') == True:
    
    retriever_docs = retriever.invoke(user_input)  
    
    st.write(retriever_docs)

