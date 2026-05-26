import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load API key from .env
load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Test it
response = llm.invoke("Say hello in one short sentence.")
print("LLM Response:", response.content)