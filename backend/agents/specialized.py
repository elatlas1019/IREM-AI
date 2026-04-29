import os
import random
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage
from .state import AgentState

class MockLLM:
    """Fallback LLM — gerçek LLM bağlanamadığında kullanılır."""
    def __init__(self, *args, **kwargs):
        pass
    def invoke(self, prompt, **kwargs):
        return type('obj', (object,), {'content': 'Bağlantı hatası, lütfen tekrar deneyin.'})()
    async def ainvoke(self, prompt, **kwargs):
        return type('obj', (object,), {'content': 'Bağlantı hatası, lütfen tekrar deneyin.'})()

# ─── Quotes library ────────────────────────────────────────────────────────────
QUOTES = {
    "Motivation": [
        {"text": "Başarı, her gün tekrarlanan küçük çabaların toplamıdır.", "author": "Robert Collier", "emoji": "🌟"},
        {"text": "Gelecek, bugün hazırlananlarındır.", "author": "Malcolm X", "emoji": "🚀"},
        {"text": "En büyük zaferimiz hiç düşmemek değil, her düştüğümüzde kalkmaktır.", "author": "Konfüçyüs", "emoji": "💪"},
        {"text": "Zorluklar, başarının değerini artıran süslerdir.", "author": "Moliere", "emoji": "💎"},
        {"text": "Eğer fırtına çıkarsa, limana yanaşma; açık denizlere açıl.", "author": "Aristoteles", "emoji": "🌊"},
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

# ─── LLM Getter (GROQ) ─────────────────────────────────────────────────────────
def get_llm(agent_type="PLAN"):
    groq_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if groq_key:
        return ChatGroq(model="llama3-8b-8192", groq_api_key=groq_key, temperature=0.7)
    else:
        return MockLLM()


# ─── Generic Node ──────────────────────────────────────────────────────────────
def generic_node(state: AgentState, agent_type: str, system_prompts: dict):
    lang = state.get("language", "tr")
    llm = get_llm(agent_type)
    if isinstance(llm, MockLLM):
        return {"messages": [AIMessage(content="⚠️ Hata: API anahtarı bulunamadı. Lütfen Streamlit ayarlarından GROQ_API_KEY ekleyin.")]}

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
        return {"messages": [AIMessage(content=f"🤖 Model Hatası: {e}")]}

    return {"messages": [response]}


# ─── Specialist Nodes ──────────────────────────────────────────────────────────
def goal_node(state: AgentState):
    prompts = {
        "tr": "Sen bir hedef belirleme koçusun. [name] için somut ve ölçülebilir hedefler listele.",
        "en": "You are a goal-setting coach. List concrete and measurable goals for [name]."
    }
    return generic_node(state, "GOAL", prompts)


def teach_node(state: AgentState):
    prompts = {
        "tr": (
            "Sen bir Türk eğitim koçusun. Kullanıcı senden bir şey istediğinde MUTLAKA tam ve eksiksiz içerik üretmelisin.\n"
            "- Test isterse soruları yaz, A B C D şıklarını yaz\n"
            "- Özet isterse detaylı özet yaz\n"
            "- Plan isterse adım adım plan yaz\n"
            "ASLA 'hazırladım' diyip içeriği gizleme. Her zaman gerçek içeriği yanıtında göster.\n"
            "Yanıtına MUTLAKA '# [Konu Başlığı]' şeklinde bir başlıkla başla."
        ),
        "en": (
            "You are an AI Education Coach. ALWAYS generate FULL content.\n"
            "- If test: write questions + options.\n"
            "- If summary: write details.\n"
            "ALWAYS start with '# [Topic Title]' header."
        ),
    }
    return generic_node(state, "TEACH", prompts)


def plan_node(state: AgentState):
    prompts = {
        "tr": "Sen bir planlama uzmanısın. Kullanıcı için detaylı bir çalışma programı hazırla. Markdown tablosu kullan.",
        "en": "You are a planning expert. Prepare a detailed study schedule using a Markdown table."
    }
    return generic_node(state, "PLAN", prompts)


def feedback_node(state: AgentState):
    prompts = {
        "tr": "Sen bir gelişim koçusun. Gelişim önerileri sun.",
        "en": "You are a growth coach. Offer improvement suggestions."
    }
    return generic_node(state, "FEEDBACK", prompts)


def health_node(state: AgentState):
    prompts = {
        "tr": "Sen bir sağlık rehberisin. Uyku ve beslenme tavsiyeleri ver.",
        "en": "You are a health guide. Give sleep and nutrition advice."
    }
    return generic_node(state, "HEALTH", prompts)


def motivation_node(state: AgentState):
    lang = state.get("language", "tr")
    sentiment = state.get("sentiment", "Neutral")
    user_name = state.get("user_name", "") or "Öğrenci"

    quote_list = QUOTES.get(sentiment, QUOTES["Neutral"])
    quote = random.choice(quote_list)

    try:
        llm = get_llm("MOTIVATION")
        if isinstance(llm, MockLLM):
            raise ValueError("No API Key")
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
        text = f"Selam {user_name}! {quote['author']} der ki: \"{quote['text']}\" {quote['emoji']} Her zaman yanındayım."
        return {"messages": [AIMessage(content=text)]}
