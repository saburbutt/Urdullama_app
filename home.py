import os
import streamlit as st
from dotenv import load_dotenv
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextStreamer

# Configuration parameters
max_seq_length = 2048  # Adjust the sequence length as needed
dtype = None  # None for auto detection; use Float16 for T4, V100; Bfloat16 for Ampere+
load_in_4bit = True  # Enable 4-bit quantization to reduce memory usage

# Load the model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    "traversaal-llm-regional-languages/Unsloth_Urdu_Llama3_1_4bit_PF100",  # Trained model
    load_in_4bit=load_in_4bit,
)
tokenizer = AutoTokenizer.from_pretrained("traversaal-llm-regional-languages/Unsloth_Urdu_Llama3_1_4bit_PF100")

# Define the chat prompt format
chat_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}
"""

# Streamlit UI setup
st.write('# UrduLlama')

# Initialize session state for past messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to display past conversations
def display_past_conversations():
    for message in st.session_state.messages:
        with st.chat_message(name=message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])

# Display past conversations when the app loads
display_past_conversations()

# Handle user input
if prompt := st.chat_input('Your message here...'):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user message
    with st.chat_message('user'):
        st.markdown(prompt)

    # Prepare the chat prompt and input tokens
    formatted_prompt = chat_prompt.format(
        "You are a chatbot. Chat in Urdu",  # Instruction
        prompt,  # User input
        "",  # Response (left blank for model generation)
    )

    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

    # Generate response with streaming
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=500)

    # Display and append AI's response (update this part based on how the model returns output)
    # ai_response = <insert code to capture model output>
    ai_response = "Sample AI response"  # Placeholder for actual model output

    with st.chat_message(name="assistant", avatar="✨"):
        st.markdown(ai_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": ai_response, "avatar": "✨"}
    )
