
from openai import OpenAI 
import streamlit as st

f = open("keys\.openai_api_key.txt")
key = f.read()
client =OpenAI(api_key= key)

st.title("Python Code Review")
prompt = st.text_input("Enter your Python code for review:")

if st.button("Review Code") == True:
  
  response = client.chat.completions.create(
          model='gpt-3.5-turbo-16k',
          messages = [
              {'role' : 'system', 'content' : """You are the helpful Assistant. Given a python code you always check for bugs in the code
                                          and provide both the original python code and the debugged code"""},
              {'role':'user','content':"prompt"}
          ]
      )

  st.write(response.choices[0].message.content)



    
   
