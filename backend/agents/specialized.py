import os
import random
import re
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage
from .state import AgentState

# ─── Quotes library ────────────────────────────────────────────────────────────
QUOTES = {
    "Motivation": [
        {"text": "Başarı, her gün tekrarlanan küçük çabaların toplamıdır.", "author": "Robert Collier", "emoji": "🌟"},
        {"text": "Gelecek, bugün hazırlananlarındır.", "author": "Malcolm X", "emoji": "🚀"},
        {"text": "En büyük zaferimiz hiç düşmemek değil, her düştüğümüzde kalkmaktır.", "author": "Konfüçyüs", "emoji": "💪"},
        {"text": "Zorluklar, başarının değerini artıran süslerdir.", "author": "Moliere", "emoji": "💎"},
        {"text": "Eğer fırtına çıkarsa, gemini limana yanaştırma; açık denizlere açıl.", "author": "Aristoteles", "emoji": "🌊"},
        {"text": "Hayat, bisiklete binmek gibidir. Dengede kalmak için hareket etmelisin.", "author": "Albert Einstein", "emoji": "🚲"},
        {"text": "Sadece güneşli günlerde yürürseniz, hedefinize asla varamazsınız.", "author": "Paulo Coelho", "emoji": "☀️"},
        {"text": "Yapabileceğinize inanın, yolun yarısını çoktan geçtiniz demektir.", "author": "Theodore Roosevelt", "emoji": "🎯"},
        {"text": "Damlaya damlaya göl olur; bugünkü küçük adımın yarınki başarın olacak.", "author": "Türk Atasözü", "emoji": "💧"},
        {"text": "Asla vazgeçme. Mucizeler her gün gerçekleşir.", "author": "H. Jackson Brown Jr.", "emoji": "✨"}
    ],
    "Tired": [
        {"text": "Dinlenmekten çekinme, ama asla vazgeçme.", "author": "İrem AI", "emoji": "🛌"},
        {"text": "Yorgunluk geçicidir, ama başarının gururu kalıcıdır.", "author": "İrem AI", "emoji": "🔋"}
    ],
    "Anxious": [
        {"text": "Derin bir nefes al. Her şey kontrol altında.", "author": "İrem AI", "emoji": "🌬️"},
        {"text": "Kaygı, gelecekteki bir problem için bugünden üzülmektir. Şimdiye odaklan.", "author": "İrem AI", "emoji": "🧘"}
    ],
    "Positive": [
        {"text": "Harika gidiyorsun! Bu enerjiyi koru.", "author": "İrem AI", "emoji": "🔥"},
        {"text": "Seninle gurur duyuyorum!", "author": "İrem AI", "emoji": "❤️"}
    ],
    "Neutral": [
        {"text": "Adım adım hedefe ilerliyoruz.", "author": "İrem AI", "emoji": "🚶"},
        {"text": "Bugün öğrenmek için harika bir gün.", "author": "İrem AI", "emoji": "📖"}
    ]
}


# ─── MockLLM (Enhanced for testing without API keys) ───────────────────────────
class MockLLM:
    def __init__(self, agent_type="PLAN"):
        self.agent_type = agent_type

    def invoke(self, messages):
        user_msg = messages[-1].content if messages else ""
        
        # Enhanced Mock Responses to avoid "empty" replies
        if self.agent_type == "TEACH":
            return AIMessage(content="# Matematik Testi (Örnek)\n\n1. 5 x 8 işleminin sonucu nedir?\nA) 35  B) 40  C) 45  D) 50\n\n2. 12 x 4 işleminin sonucu nedir?\nA) 44  B) 46  C) 48  D) 50\n\nBu bir demo yanıtıdır. Lütfen tam içerik için API anahtarınızı ekleyin.")
        elif self.agent_type == "PLAN":
            return AIMessage(content="# Çalışma Planı\n\n| Gün | Saat | Konu | Aktivite |\n|---|---|---|---|\n| Pazartesi | 18:00 | Matematik | Soru Çözümü |\n| Salı | 19:00 | Fen | Konu Tekrarı |")
        
        return AIMessage(content="Sana yardımcı olmak için buradayım! Lütfen API anahtarınızı (ANTHROPIC_API_KEY) kontrol edin.")


# ─── LLM Getter ────────────────────────────────────────────────────────────────
def get_llm(agent_type="PLAN"):
    if os.getenv("ANTHROPIC_API_KEY"):
        return ChatAnthropic(model="claude-sonnet-4-20250514")
    elif os.getenv("GROQ_API_KEY"):
        return ChatGroq(model="llama-3.3-70b-versatile")
    elif os.getenv("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    else:
        return MockLLM(agent_type)


# ─── Generic Node ──────────────────────────────────────────────────────────────
def generic_node(state: AgentState, agent_type: str, system_prompts: dict):
    lang = state.get("language", "tr")
    llm = get_llm(agent_type)

    system_msg = system_prompts.get(lang, system_prompts.get("en", ""))
    user_name = state.get("user_name", "") or "Öğrenci"
    system_msg = system_msg.replace("[name]", user_name)
    
    lang_instruction = (
        "MUTLAKA Türkçe yanıt ver. İngilizce kullanma." if lang == "tr"
        else "ALWAYS reply in English. Do not use Turkish."
    )
    system_msg = f"{lang_instruction}\n\n{system_msg}"

    messages = [SystemMessage(content=system_msg)] + state["messages"]

    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"LLM Error in {agent_type}: {e}")
        fallback_llm = MockLLM(agent_type)
        response = fallback_llm.invoke(messages)

    return {"messages": [response]}


# ─── Specialist Nodes ──────────────────────────────────────────────────────────
def goal_node(state: AgentState):
    prompts = {
        "tr": (
            "Sen bir hedef belirleme koçusun. [name] için gerçekçi hedefler belirle. "
            "ASLA boş cevap verme, her zaman somut hedefleri listele."
        ),
        "en": (
            "You are a goal-setting coach. Set realistic goals for [name]. "
            "NEVER give empty replies, always list concrete goals."
        ),
    }
    return generic_node(state, "GOAL", prompts)


def teach_node(state: AgentState):
    prompts = {
        "tr": (
            "Sen bir Türk eğitim koçusun. Kullanıcı senden bir şey istediğinde MUTLAKA tam ve eksiksiz içerik üretmelisin.\n"
            "- Test isterse soruları yaz, A B C D şıklarını yaz\n"
            "- Özet isterse detaylı özet yaz\n"
            "- Plan isterse adım adım plan yaz\n"
            "ASLA 'hazırladım, işte sorular' gibi boş cevap verme. Her zaman gerçek içeriği üret ve yaz.\n"
            "Yanıtına MUTLAKA '# [Konu Başlığı]' şeklinde bir başlıkla başla."
        ),
        "en": (
            "You are an AI Education Coach. When the user asks for something, you MUST generate FULL and COMPLETE content.\n"
            "- If they want a test, write the questions and A B C D options.\n"
            "- If they want a summary, write a detailed summary.\n"
            "- If they want a plan, write a step-by-step plan.\n"
            "NEVER give empty replies like 'I prepared them'. Always generate and write the actual content.\n"
            "ALWAYS start with a '# [Topic Title]' header."
        ),
    }
    return generic_node(state, "TEACH", prompts)


def plan_node(state: AgentState):
    prompts = {
        "tr": (
            "Sen bir eğitim planlama uzmanısın. Kullanıcı için MUTLAKA tam bir çalışma programı hazırla.\n"
            "Programı Markdown tablosu formatında sun (Gün | Saat | Konu | Aktivite).\n"
            "ASLA boş veya kısa cevap verme. Gerçek ve detaylı bir plan üret."
        ),
        "en": (
            "You are an education planning expert. You MUST prepare a full study program for the user.\n"
            "Present the program in Markdown table format (Day | Time | Subject | Activity).\n"
            "NEVER give empty or short replies. Generate a real and detailed plan."
        ),
    }
    return generic_node(state, "PLAN", prompts)


def feedback_node(state: AgentState):
    prompts = {
        "tr": "Sen bir gelişim koçusun. Sınav sonuçlarını detaylıca analiz et ve iyileştirme önerileri sun.",
        "en": "You are a growth coach. Analyze exam results in detail and offer improvement suggestions."
    }
    return generic_node(state, "FEEDBACK", prompts)


def health_node(state: AgentState):
    prompts = {
        "tr": "Sen bir sağlık rehberisin. Uyku, beslenme ve stres yönetimi için gerçekçi tavsiyeler ver.",
        "en": "You are a health guide. Give realistic advice for sleep, nutrition, and stress management."
    }
    return generic_node(state, "HEALTH", prompts)


def motivation_node(state: AgentState):
    lang = state.get("language", "tr")
    sentiment = state.get("sentiment", "Neutral")
    user_name = state.get("user_name", "") or "Öğrenci"

    quote_list = QUOTES.get(sentiment, QUOTES["Neutral"])
    quote = random.choice(quote_list)

    llm = get_llm("MOTIVATION")
    if not isinstance(llm, MockLLM):
        try:
            lang_instruction = "MUTLAKA Türkçe yanıt ver." if lang == "tr" else "ALWAYS reply in English."
            system_prompt = (
                f"{lang_instruction}\n"
                f"You are a warm AI coach. User: {user_name}, State: {sentiment}.\n"
                f"Quote: '{quote['text']}' by {quote['author']}.\n"
                f"Be very supportive and personal. 3-4 sentences max."
            )
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = llm.invoke(messages)
            return {"messages": [response]}
        except:
            pass

    # Fallback
    text = f"Selam {user_name}! {quote['author']} der ki: \"{quote['text']}\" {quote['emoji']} Yanındayım!"
    return {"messages": [AIMessage(content=text)]}
