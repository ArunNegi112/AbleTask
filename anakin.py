import os 
from getpass import getpass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
try :
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as error:
    print("Env variables not found")
# --- Initialize model ---
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

# --- Simple memory storage per session (simulate memory per user id) ---
store = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Build runnable chain
# It will fill the prompt template and call the LLM itself when we invoke the model
chain = prompt | llm

# Add memory to the chain
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
def run_chatbot(user_input, session_id="user1"):
    response = chatbot.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content
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
# Create the second prompt template and chain
prompt_2 = PromptTemplate.from_template(template_2)
chain_2 = prompt_2 | llm_2

def parse_tasks_to_json(task_breakdown):
    """
    Takes the formatted task breakdown from the first model
    and returns a clean JSON using the second model
    """
    response = chain_2.invoke({"input": task_breakdown})
    return response.content

# Example of using both models in sequence
def get_parsed_tasks(user_input, session_id="user1"):
    # First get the formatted task breakdown
    task_breakdown = run_chatbot(user_input, session_id)
    # Then parse it to JSON
    json_tasks = parse_tasks_to_json(task_breakdown)
    return json_tasks

# Test the combined models
test_input = "Help me break down studying for a math exam in 2 hours"
parsed_result = get_parsed_tasks(test_input)
print(parsed_result)

