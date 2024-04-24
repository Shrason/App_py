import google.generativeai as genai
import streamlit as st

f = open('keys\gemini_api_key.txt')
key = f.read()

genai.configure(api_key=key)

model = genai.GenerativeModel(model_name='gemini-1.5-pro-latest', 
                              system_instruction="""You are the helpful Assistant. Given a python code you always check for bugs in the code
                                          and provide both the original python code and the debugged code""")

st.title('Review App')

user_prompt = st.text_area("Enter your Python code for review:")

if st.button("Review Code") == True:
    response = model.generate_content(user_prompt)

    st.write(response.text)