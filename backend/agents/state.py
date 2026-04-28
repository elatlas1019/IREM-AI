from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    next_agent: str
    user_context: str
    sentiment: str  # Positive, Negative, Neutral, Anxious, Tired
    energy_score: int  # 1-10
    user_name: str
    language: str  # "tr" or "en"
