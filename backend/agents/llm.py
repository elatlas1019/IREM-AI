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
    use_gemini = os.getenv("USE_GEMINI", "False").lower() in ("true", "1", "t")
    google_key = os.getenv("GOOGLE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    try:
        # Prioritize Groq if available as it's very fast and reliable for this environment
        if is_valid_key(groq_key) and not use_gemini:
            from langchain_groq import ChatGroq
            return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
        elif use_gemini and is_valid_key(google_key):
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Use gemini-1.5-flash as it is more likely to be available globally
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
        elif is_valid_key(anthropic_key):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model_name="claude-3-5-sonnet-20241022", temperature=0.7)
        elif is_valid_key(groq_key): # Fallback to Groq if others not specifically requested
            from langchain_groq import ChatGroq
            return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
        else:
            print("Using Dummy LLM (No valid API key found)")
            return DummyChatLLM()
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return DummyChatLLM()

def get_json_llm():
    use_gemini = os.getenv("USE_GEMINI", "False").lower() in ("true", "1", "t")
    google_key = os.getenv("GOOGLE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    try:
        if is_valid_key(groq_key) and not use_gemini:
            from langchain_groq import ChatGroq
            return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        elif use_gemini and is_valid_key(google_key):
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        elif is_valid_key(groq_key):
            from langchain_groq import ChatGroq
            return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        elif is_valid_key(anthropic_key):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model_name="claude-3-5-sonnet-20241022", temperature=0)
        else:
            return DummyLLM()
    except Exception as e:
        print(f"Error initializing JSON LLM: {e}")
        return DummyLLM()
