import google.generativeai as genai
import streamlit as st

f = open('keys\gemini_api_key.txt')
key = f.read()

genai.configure(api_key=key)

model = genai.GenerativeModel(model_name='gemini-1.5-pro-latest', 
                              system_instruction="""You are the helpful AI Teaching Assistant. Given a Data Science topic you always helps the user uderstand the topic in simple way.If the question is not related to Data Science topic, response with 'This is beyond my knowledge' """)

st.title('AI Data Science Chatbot')

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
