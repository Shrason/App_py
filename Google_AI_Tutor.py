import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Configure Gemini with API key
#api_key = os.environ.get("GEMINI_API_KEY")
#if api_key is None:
    #raise ValueError("API key not found. Please check your .env file.")


# Configure Gemini with API key
genai.configure(api_key="AIzaSyC8ycTAuxvD9gBSgVn3UQwmD65yuQkOq88")

# Define generation configuration
generation_config = {
    "temperature": 0.85,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Create the model instance
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", generation_config=generation_config, 
    system_instruction="You are a tutor, who explains the topic systematically from basic to advance thus clearing the concepts in easy way. The tutor provides simple examples and applications of topic for further clear understanding. On instructed the tutor can set daily learning targets. The tutor also check and give feedback on the understanding of the learner by asking relevent application question and/or giving task to solve. ",
)

# Initialize chat session with empty history
chat_session = model.start_chat(history=[])

# Session state for storing chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Title and description for the app
st.title("Chat with Gemini Tutor")
st.markdown("Welcome to the Gemini Tutor app. Ask any question and get a response from the Gemini AI model.")

# Chat input field
user_input = st.text_input("Ask your question:", key="user_input")

# Button to send the message
if st.button("Send"):
    # Send message to Gemini and get response
    response = chat_session.send_message(user_input)
    st.session_state["chat_history"].append({"user": user_input, "response": response.text})

# Display chat history
for message in st.session_state["chat_history"]:
    if message["user"] == user_input:
        st.write("You:", message["user"])
    else:
        st.write("Tutor:", message["response"])

# Footer
st.markdown("---")
st.markdown("Powered by Gemini AI")
