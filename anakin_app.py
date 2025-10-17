import streamlit as st
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

# Page config
st.set_page_config(page_title="Anakin - Task Breakdown Assistant", page_icon="🤖")

# Initialize session state for storing chat history
if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Initialize models
@st.cache_resource
def init_models():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
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
- If user specifies a time range, ensure the total estimated time for all microtasks falls within that range.
- keep the total number of microtasks between 3 to 5 if time is between 1 to 2 hours, increase number of tasks to 5 to 12 if time is between 4-5 unless the task is very complex, And you can keep increase those numbers if times increase in the same manner. 

Chat history:
{chat_history}

Here is the user's task to break down:
{input}
"""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    
    chatbot = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    # Second model for parsing
    llm_2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    template_2 = """
You are a task parser that converts formatted task lists into JSON format.
Your input will be the output from another AI that creates task breakdowns with emojis, formatting, and time estimates.

Your job is to:
1. Extract only the tasks and their time estimates (for each task respectively)
2. Return a clean JSON with two fields:
   - Tasks (numbered as "Task 1", "Task 2", etc.)
   - Time (numbered as "Time require T1", "Time required T2", etc. )

Remove all formatting, emojis, and extra text. Just extract the core task descriptions and times.

Example input:
📋 Breaking down: Study for Math Exam

**Step 1: 📚 Review Chapter Notes**
Organize and read through class notes
⏱️ **Time:** 30 minutes

**Step 2: 🎯 Practice Problems**
Work through end-of-chapter exercises
⏱️ **Time:** 45 minutes

✨ You've got this! Take it one step at a time.

Expected output:
{{
    "Task 1": "Review Chapter Notes",
    "Time required T1": "20 minutes"
    "Task 2": "Practice Problems",
    "Time required T2": "75 minutes"
}}

Here is the task breakdown to parse:
{input}
"""
    
    prompt_2 = PromptTemplate.from_template(template_2)
    chain_2 = prompt_2 | llm_2
    
    return chatbot, chain_2

# Main app
st.title("🤖 Hey, This is Anakin, So what are we doing today?")
st.markdown("---")

# Check for API key
load_dotenv()

# Initialize models
try:
    chatbot, chain_2 = init_models()
except Exception as e:
    st.error(f"Error initializing models: {e}")
    st.stop()

# User input
user_input = st.text_area(
    "What task would you like to break down?",
    placeholder="e.g., Study for Math exam in 2 hours",
    height=100
)

if st.button("Break it down! 🚀", type="primary"):
    if user_input:
        with st.spinner("Anakin is thinking..."):
            try:
                # Get formatted breakdown
                task_breakdown = chatbot.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "user1"}},
                ).content
                
                # Parse to JSON
                json_response = chain_2.invoke({"input": task_breakdown}).content
                
                # Clean JSON response (remove markdown code blocks if present)
                json_str = json_response.strip()
                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                if json_str.startswith("```"):
                    json_str = json_str[3:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                json_str = json_str.strip()
                
                # Parse JSON
                tasks_data = json.loads(json_str)
                
                # Display results
                st.markdown("### 📝 Your Task Breakdown")
                st.markdown("---")
                
                # Extract and display tasks
                task_num = 1
                while f"Task {task_num}" in tasks_data:
                    task_key = f"Task {task_num}"
                    time_key = f"Time required T{task_num}"
                    
                    task = tasks_data.get(task_key, "N/A")
                    time_required = tasks_data.get(time_key, "N/A")
                    
                    st.markdown(f"**Task {task_num}:** {task}")
                    st.markdown(f"**Time required:** {time_required}")
                    st.markdown("")
                    
                    task_num += 1
                
                if task_num == 1:
                    st.info("No tasks were parsed. Here's the original breakdown:")
                    st.markdown(task_breakdown)
                
            except json.JSONDecodeError:
                st.error("Failed to parse tasks. Here's the raw breakdown:")
                st.markdown(task_breakdown)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a task to break down.")

# Sidebar with info
with st.sidebar:
    st.markdown("### About Anakin")
    st.markdown("""
    Anakin is an AI assistant designed to help neurodivergent individuals 
    break down tasks into manageable microtasks.
    
    **Tips for best results:**
    - Include estimated time (e.g., "in 2 hours")
    - Be specific about your task
    - Start with smaller goals if unsure
    """)
    
    if st.button("Clear History"):
        st.session_state.store = {}
        st.success("Chat history cleared!")
        st.rerun()