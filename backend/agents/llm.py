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
        return AIMessage(content="[Mocked Agent Response] I am a mocked AI agent. Please configure your API keys in the backend/.env file to get real responses.")

def is_valid_key(key):
    return key and key.strip() and "your_key" not in key.lower() and "api_key" not in key.lower()

def get_llm():
    use_gemini = True # Force Gemini as per request
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    try:
        if is_valid_key(gemini_key):
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=gemini_key)
        elif is_valid_key(groq_key):
            from langchain_groq import ChatGroq
            return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
        else:
            print("Using Dummy LLM (No valid API key found)")
            return DummyChatLLM()
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return DummyChatLLM()

def get_json_llm():
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    try:
        if is_valid_key(gemini_key):
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=gemini_key)
        elif is_valid_key(groq_key):
            from langchain_groq import ChatGroq
            return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        else:
            return DummyLLM()
    except Exception as e:
        print(f"Error initializing JSON LLM: {e}")
        return DummyLLM()
