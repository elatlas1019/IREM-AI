import re
import json
import os
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState
from agents.specialized import get_llm


def detect_language(text: str) -> str:
    """Robustly detect Turkish vs English from message content."""
    # Turkish-specific characters
    tr_chars = set("çğışöüÇĞİŞÖÜ")
    if any(ch in tr_chars for ch in text):
        return "tr"
    # Turkish common words (lowercase check)
    text_lower = text.lower()
    tr_words = [
        "merhaba", "nasılsın", "günaydın", "iyi", "yardım", "heyecan",
        "bugün", "çok", "ama", "için", "bana", "benim", "ders", "okul",
        "sınav", "hedef", "plan", "program", "motivasyon", "teşekkür",
        "tamam", "evet", "hayır", "anlamıyorum", "anlat", "nasıl", "ne",
        "kim", "neden", "artık", "sadece", "çünkü", "fakat", "da", "de",
        "bir", "ve", "ile", "sen", "ben", "biz", "onlar", "onu", "seni"
    ]
    if any(word in text_lower for word in tr_words):
        return "tr"
    return "en"


def orchestrator_node(state: AgentState):
    messages = state.get("messages", [])
    if not messages:
        return {"next_agent": "MOTIVATION", "language": "tr", "sentiment": "Neutral", "energy_score": 6}

    last_message = messages[-1].content
    last_message_lower = last_message.lower()

    # 1. Detect Language FIRST — used both for LLM prompt and fallback
    language = detect_language(last_message)

    # Extract user name from context if available
    user_context = state.get("user_context", "")
    user_name = ""
    if "name:" in user_context.lower():
        for part in user_context.split(","):
            if "name:" in part.lower():
                user_name = part.split(":")[-1].strip()

    # 2. Try LLM-based Orchestration if key is available
    llm = get_llm("ORCHESTRATOR")
    from agents.specialized import MockLLM

    if not isinstance(llm, MockLLM):
        try:
            system_prompt = f"""You are the orchestrator for an AI Learning Coach system.
The user wrote in: {'Turkish' if language == 'tr' else 'English'}.
Your job is to analyze the user message and return a JSON object with:
1. "next_agent": One of ["GOAL", "PLAN", "FEEDBACK", "MOTIVATION", "HEALTH", "TEACH"]
2. "sentiment": One of ["Positive", "Neutral", "Tired", "Anxious"]
3. "energy_score": integer 1-10
4. "language": "{language}" (ALWAYS use this exact value)

Guidelines:
- Expressing feelings only (excited, happy, sad, tired) → MOTIVATION
- Setting a goal → GOAL
- Study schedule/routine request → PLAN
- Discussing grades/exam results → FEEDBACK
- Learning a topic, explanation, quiz → TEACH
- Sleep, food, health, exercise → HEALTH
- Short greeting with no task → MOTIVATION
- Default when unclear → MOTIVATION (not PLAN)

Return ONLY the JSON. Example: {{"next_agent": "MOTIVATION", "sentiment": "Positive", "energy_score": 8, "language": "{language}"}}"""

            response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_message)])
            res_content = response.content.strip()
            if "```json" in res_content:
                res_content = res_content.split("```json")[1].split("```")[0].strip()
            elif "```" in res_content:
                res_content = res_content.split("```")[1].split("```")[0].strip()

            data = json.loads(res_content)
            # Force language to what we detected — never trust LLM to override
            return {
                "language": language,
                "sentiment": data.get("sentiment", "Neutral"),
                "energy_score": data.get("energy_score", 6),
                "next_agent": data.get("next_agent", "MOTIVATION"),
                "user_name": user_name,
            }
        except Exception as e:
            print(f"Orchestrator LLM Error: {e}. Falling back to heuristics.")

    # 3. Heuristic Fallback
    sentiment = "Neutral"
    energy_score = 6

    tired_words = [
        "yorgun", "bunaldım", "bıktım", "sıkıldım", "yorucu", "bitik", "enerjim yok",
        "tired", "exhausted", "overwhelmed", "drained", "can't", "cannot"
    ]
    anxious_words = [
        "endişe", "korku", "panik", "stres", "kaygı", "korkuyorum", "gerginim",
        "anxious", "scared", "panic", "stress", "worry", "nervous", "afraid"
    ]
    positive_words = [
        "harika", "mükemmel", "başardım", "mutlu", "sevinçli", "heyecanlı", "heyecan",
        "süper", "muhteşem", "great", "amazing", "awesome", "happy", "excited", "excellent"
    ]

    if any(word in last_message_lower for word in tired_words):
        sentiment = "Tired"
        energy_score = 3
    elif any(word in last_message_lower for word in anxious_words):
        sentiment = "Anxious"
        energy_score = 4
    elif any(word in last_message_lower for word in positive_words):
        sentiment = "Positive"
        energy_score = 8

    # Routing
    next_agent = "MOTIVATION"  # Better default than PLAN

    goal_words = [
        "hedef", "amaç", "goal", "target", "want to achieve",
        "olmak istiyorum", "neyi başarmak", "hedefim", "hayalim"
    ]
    plan_words = [
        "plan", "program", "schedule", "çizelge", "study plan",
        "nasıl çalışmalıyım", "ders programı", "çalışma planı", "takvim oluştur"
    ]
    feedback_words = [
        "sınav", "exam", "grade", "puan", "başarı", "result", "sonuç",
        "nasıl geçti", "deneme", "test sonucu", "notlarım", "kaç aldım"
    ]
    health_words = [
        "uyku", "sleep", "yemek", "eat", "sağlık", "health",
        "spor", "exercise", "nefes", "su", "water", "beslenme", "vitamin"
    ]
    teach_words = [
        "soru", "özet", "anlat", "nedir", "kimdir", "nasıl olur",
        "açıkla", "test hazırla", "konu anlat", "question", "summary",
        "explain", "describe", "what is", "how does", "who is", "quiz"
    ]

    if any(word in last_message_lower for word in teach_words):
        next_agent = "TEACH"
    elif any(word in last_message_lower for word in goal_words):
        next_agent = "GOAL"
    elif any(word in last_message_lower for word in feedback_words):
        next_agent = "FEEDBACK"
    elif any(word in last_message_lower for word in plan_words):
        next_agent = "PLAN"
    elif any(word in last_message_lower for word in health_words):
        next_agent = "HEALTH"
    elif any(word in last_message_lower for word in positive_words + tired_words + anxious_words):
        next_agent = "MOTIVATION"
    elif sentiment in ["Tired", "Anxious", "Positive"]:
        next_agent = "MOTIVATION"

    return {
        "language": language,
        "sentiment": sentiment,
        "energy_score": energy_score,
        "next_agent": next_agent,
        "user_name": user_name,
    }
