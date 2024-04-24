import google.generativeai as genai
import streamlit as st

f = open('keys\gemini_api_key.txt')
key = f.read()

genai.configure(api_key=key)

model = genai.GenerativeModel(model_name='gemini-1.5-pro-latest', 
                              system_instruction="""You are the helpful AI Assistant. Given a query you always helps with the relative query topic""")

st.title('AI chatbot')

if 'chat_history' not in st.session_state:
    st.session_state['chat_history']=[]
    
chat = model.start_chat(history=st.session_state['chat_history'])

for msg in chat.history:
    st.chat_message(msg.role).write(msg.parts[0].text)
    
user_prompt = st.chat_input()
    
if user_prompt:
    st.chat_message('user').write(user_prompt)
    response = chat.send_message(user_prompt)
    st.chat_message('ai').write(response.text)
    st.session_state['chat_history'] = chat.history