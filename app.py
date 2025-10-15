import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

# Load environment variables from .env if available
load_dotenv()

# Sidebar inputs for API keys and project name
if "LANGSMITH_API_KEY" not in os.environ:
    langsmith_key = st.sidebar.text_input("Langsmith API Key:", key="langsmith_api")
    if langsmith_key:
        os.environ['LANGSMITH_API_KEY'] = langsmith_key

if "LANGSMITH_PROJECT" not in os.environ:
    project_name = st.sidebar.text_input("Langsmith Project Name:", value="default", key="langsmith_project")
    os.environ['LANGSMITH_PROJECT'] = project_name if project_name else "default"

if "GOOGLE_API_KEY" not in os.environ:
    google_key = st.sidebar.text_input("Google API Key:", key="google_api", type="password")
    if google_key:
        os.environ['GOOGLE_API_KEY'] = google_key

# Enable Langsmith tracing
os.environ['LANGSMITH_TRACING'] = 'true'

# Initialize Google LLM only if API key is available
if os.environ.get('GOOGLE_API_KEY'):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
else:
    llm = None

# Define prompt template with improved formatting instructions
template = """
You are Anakin, an AI assistant designed to help neurodivergent individuals who struggle with starting tasks and maintaining focus.
Your purpose is to increase their productivity by breaking down any given learning or productivity-related task into clear, manageable, neuro-optimized microtasks.

Guidelines:
- Always respond with a neutral tone.
- Only output a step-by-step list of microtasks relevant to the input task.
- Each microtask should be concise, actionable, and include a relevant emoji to enhance clarity and engagement.
- Adapt the number and granularity of microtasks based on the task's length and difficulty. Larger or more complex tasks should have more detailed microtasks.
- Never provide commentary or advice outside of the microtask list, except when the task is unrealistic.
- If the task is unrealistic (e.g., "Learn full stack web development in one day"), respond with a neutral message indicating it may not be achievable as stated and suggest revising it.
- If the user does not specify an estimated time range for the task, ask them to provide how much time they expect to spend on it so you can optimize the microtasks more accurately.
- If the user asks for or implies wanting to know how much time to allocate to each microtask, provide a time estimate for each microtask in the step-by-step list, with each time estimate adding up to the total estimated time.

FORMATTING REQUIREMENTS:
- Start with a brief header like "📋 Breaking down: [Task Name]" followed by a blank line
- Present microtasks as a numbered list with clear spacing
- Use this format for each step:
  **Step [number]: [Action with emoji]**
  Brief description if needed
  ⏱️ **Time:** [X minutes] (if time estimates requested)
  
- Add helpful section breaks for complex tasks (e.g., "### 🎯 Phase 1: Setup")
- End with an encouraging closing line like "✨ You've got this! Take it one step at a time."
- Use markdown formatting to make the output visually appealing and easy to scan

Chat history:
{chat_history}

Here is the user's task to break down:
{input}
"""

prompt = PromptTemplate(template=template, input_variables=["input", "chat_history"])

# Simple memory per session
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Streamlit app config
st.set_page_config(page_title="Study Checkpoint Chatbot", page_icon="💡")
st.markdown(
    """
    <style>
    body { background-color: #F0FFF0; }
    .stTextInput input { background-color: #F5FFFA; color: #333; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = "user1"
if "conversation" not in st.session_state:
    st.session_state.conversation = []

st.title("Study Checkpoint Chatbot")
st.subheader("This is Anakin, what are we doing today?")

# Check if API key is set
if not os.environ.get('GOOGLE_API_KEY'):
    st.warning("⚠️ Please enter your Google API Key in the sidebar to start chatting.")
else:
    # Chat input with custom placeholder
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your Task", value="", key="user_input", placeholder="So what are we doing today?")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})
        
        # Get chat history
        history = get_session_history(st.session_state.session_id)
        chat_history_text = "\n".join([m.content for m in history.messages]) if history.messages else ""
        
        # Build prompt with chat history
        prompt_text = prompt.format(input=user_input, chat_history=chat_history_text)
        
        # Generate response
        try:
            response = llm.invoke(prompt_text)
            bot_reply = response.content
            
            # Update session history
            history.add_user_message(user_input)
            history.add_ai_message(bot_reply)
            
            st.session_state.conversation.append({"role": "assistant", "content": bot_reply})
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            bot_reply = "Sorry, I encountered an error. Please try again."
            st.session_state.conversation.append({"role": "assistant", "content": bot_reply})
        
        st.rerun()

# Display conversation with better formatting
for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown("**🤖 Anakin:**")
        st.markdown(msg['content'])
        st.markdown("---")  # Add separator between responses