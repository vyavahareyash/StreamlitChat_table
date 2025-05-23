import streamlit as st
from datetime import datetime
import os
from chat_backend import process_user_question, create_user_message

# Page configuration
st.set_page_config(
    page_title="Database Chatbot",
    page_icon="💬",
    layout="wide",
)

# Read and apply custom CSS
css_file_path = os.path.join(os.path.dirname(__file__), 'static', 'styles.css')
try:
    with open(css_file_path, 'r') as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    print(f"Warning: CSS file not found at {css_file_path}")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Title and description
st.title("Database Chatbot")
st.subheader("Ask questions about your data")

# Sidebar for settings
with st.sidebar:
    st.header("About")
    st.write("This chatbot allows you to query your database using natural language.")
    
    st.header("Settings")
    show_timestamps = st.checkbox("Show timestamps", value=True)
    
    st.header("Sample Questions")
    st.markdown("- What are the top selling products?")
    st.markdown("- How many customers do we have in California?")
    st.markdown("- What was the revenue in Q1 2024?")
    
    # You could add more settings here like:
    # - Database connection settings
    # - Model parameters
    # - Theme selection


# Chat container
st.markdown("<h3>Chat</h3>", unsafe_allow_html=True)

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-container">
                <div class="user-message">
                    <b>You:</b> {message["content"]}
                </div>
                {f'<div class="timestamp">{message["timestamp"]}</div>' if show_timestamps else ''}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-container">
                <div class="bot-message">
                    <b>Bot:</b> {message["content"]}
                </div>
                {f'<div class="timestamp">{message["timestamp"]}</div>' if show_timestamps else ''}
            """, unsafe_allow_html=True)
            
            # Create expandable sections for additional details
            with st.expander("View Details"):
                tabs = st.tabs(["Prompt", "Data", "Query"])
                with tabs[0]:
                    st.code(message.get("prompt", "No prompt information available"))
                with tabs[1]:
                    if "data" in message and message["data"] is not None:
                        st.dataframe(message["data"])
                    else:
                        st.info("No data used for this response")
                with tabs[2]:
                    st.code(message.get("query", "No query information available"))
            
            st.markdown("</div>", unsafe_allow_html=True)

# Input area for new questions
st.markdown("<h3>Ask a question</h3>", unsafe_allow_html=True)

# Initialize session state for form
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False


# Create a form for the input
with st.form(key="question_form", clear_on_submit=True):
    user_question = st.text_input("Type your question here...", key="user_input")
    submit_button = st.form_submit_button("Ask")
    
    if submit_button:
        st.session_state.form_submitted = True

# Process the user input
if st.session_state.form_submitted and user_question:
    # Step 1: Add user message immediately
    user_msg = create_user_message(user_question)
    st.session_state.chat_history.append(user_msg)

    # Step 2: Reset form state
    st.session_state.form_submitted = False

    # Step 3: Trigger rerun to show user message immediately
    st.rerun()

# If last message was from the user and no bot response yet
if (
    st.session_state.chat_history and
    st.session_state.chat_history[-1]["role"] == "user" and
    (
        len(st.session_state.chat_history) == 1 or
        st.session_state.chat_history[-2]["role"] != "bot"
    )
):
    # Show loading spinner while generating bot response
    with st.spinner("Bot is thinking..."):
        user_question = st.session_state.chat_history[-1]["content"]
        bot_response = process_user_question(user_question)
        st.session_state.chat_history.append(bot_response)
        st.rerun()



# Add a footer
st.markdown("---")
st.markdown("<center>© 2025 | Database Chatbot</center>", unsafe_allow_html=True)