import os
import streamlit as st
from together import Together
from dotenv import load_dotenv


Together_API_KEY: st.secrets["TOGETHER_API_KEY"]
# Setup the client using Together API key
client = Together(api_key=Together_API_KEY)

MODEL_ROLE = 'assistant'  # Correct role for the AI responses
AI_AVATAR_ICON = '✨'

st.write('# UrduLlama')

# Initialize session state to store past messages if not already done
if 'messages' not in st.session_state:
    st.session_state.messages = []  # Initialize an empty list to store past conversation

# Function to display past conversations in the UI
def display_past_conversations():
    for message in st.session_state.messages:
        with st.chat_message(
            name=message['role'],
            avatar=message.get('avatar'),
        ):
            st.markdown(message['content'])

# Display past conversation only once at the start
display_past_conversations()

# React to user input
if prompt := st.chat_input('Your message here...'):

    # Add user message to session state (conversation history)
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)

    # Prepare the conversation history for the AI (past context)
    ai_input_messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]

    # Send conversation (including past context) to the AI
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        messages=ai_input_messages  # Sending the full conversation history
    )

    # Extract the AI's response
    ai_response = response.choices[0].message.content

    # Display AI's response in chat message container
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        st.markdown(ai_response)

    # Add AI's response to session state (conversation history)
    st.session_state.messages.append(
        {
            "role": MODEL_ROLE,  # Correct role for AI responses is 'assistant'
            "content": ai_response,
            "avatar": AI_AVATAR_ICON
        }
    )
