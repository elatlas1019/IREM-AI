import os
from dotenv import load_dotenv

load_dotenv()

class DummyLLM:
    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content='{"agent": "MOTIVATION", "summary": "test"}')

class DummyChatLLM:
    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        return AIMessage(content="[Mocked Agent Response] GROQ_API_KEY bulunamadı. Lütfen .env veya Streamlit Secrets'a ekleyin.")

def is_valid_key(key):
    return key and key.strip() and "your_key" not in key.lower() and "api_key" not in key.lower()

def get_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    try:
        if is_valid_key(groq_key):
            from langchain_groq import ChatGroq
            return ChatGroq(model_name="llama3-8b-8192", temperature=0.7)
        else:
            print("Using Dummy LLM (No valid GROQ_API_KEY found)")
            return DummyChatLLM()
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return DummyChatLLM()

def get_json_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    try:
        if is_valid_key(groq_key):
            from langchain_groq import ChatGroq
            return ChatGroq(model_name="llama3-8b-8192", temperature=0)
        else:
            return DummyLLM()
    except Exception as e:
        print(f"Error initializing JSON LLM: {e}")
        return DummyLLM()
