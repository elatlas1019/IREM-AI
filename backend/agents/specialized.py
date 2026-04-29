import os
import random
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage
from .state import AgentState

# ─── Quotes library ────────────────────────────────────────────────────────────
QUOTES = {
    "Tired": [
        {"text": "Zorluklarla karşılaştığında, içindeki gücü hatırla. Güçsüz biri bu kadar yol katamazdı.", "author": "Marcus Aurelius", "emoji": "🌱"},
        {"text": "Rest when you must, but never quit. The body tires; the will does not.", "author": "Epictetus", "emoji": "💫"},
        {"text": "Yorgunluk, büyümenin bedelidir. Ağaçlar da fırtınada kök salar.", "author": "Rumi", "emoji": "🌊"},
        {"text": "Between stimulus and response, there is a space. In that space is your power.", "author": "Viktor Frankl", "emoji": "🕊️"},
        {"text": "Güneş her gün doğar. Sen de her gün yeniden başlayabilirsin.", "author": "Seneca", "emoji": "🌅"},
        {"text": "Even the darkest night will end and the sun will rise.", "author": "Victor Hugo", "emoji": "🌤️"},
        {"text": "Düşmek başarısızlık değildir. Düştüğün yerde kalmak başarısızlıktır.", "author": "Mary Pickford", "emoji": "🔄"},
        {"text": "It does not matter how slowly you go as long as you do not stop.", "author": "Confucius", "emoji": "🐢"},
    ],
    "Anxious": [
        {"text": "Kaygı, henüz yaşanmamış acıların bedelini peşin ödemektir.", "author": "Seneca", "emoji": "🧘"},
        {"text": "You have power over your mind, not outside events. Realize this and you will find strength.", "author": "Marcus Aurelius", "emoji": "🌸"},
        {"text": "Gelecekten korkan, şimdiki anı kaybeder. Şimdiki ana odaklan.", "author": "Epictetus", "emoji": "✨"},
        {"text": "Life is 10% what happens to you and 90% how you react to it.", "author": "Charles R. Swindoll", "emoji": "🎯"},
        {"text": "Endişe bir sandalye gibidir — seni meşgul eder ama hiçbir yere götürmez.", "author": "Erma Bombeck", "emoji": "🪑"},
        {"text": "Everything you've ever wanted is on the other side of fear.", "author": "George Addair", "emoji": "🚪"},
        {"text": "Nefes al. Bu an geçecek. Her geçen an seni daha güçlü yapıyor.", "author": "Rumi", "emoji": "🌬️"},
        {"text": "Courage is not the absence of fear, but acting in spite of it.", "author": "Mark Twain", "emoji": "⚔️"},
    ],
    "Positive": [
        {"text": "Harika gidiyorsun! Momentum'u koru — büyük başarılar böyle inşa edilir.", "author": "Isaac Newton", "emoji": "🔥"},
        {"text": "Success is not final, failure is not fatal: it is the courage to continue that counts.", "author": "Winston Churchill", "emoji": "⚡"},
        {"text": "Bu enerjiyle dağları aşarsın. Hedefine odaklan.", "author": "Marie Curie", "emoji": "🌟"},
        {"text": "The more I learn, the more I realize how much I don't know — and that's exciting!", "author": "Albert Einstein", "emoji": "🚀"},
        {"text": "Başardığın her şey, bir sonraki başarının temelidir.", "author": "Oprah Winfrey", "emoji": "🏆"},
        {"text": "Education is the most powerful weapon you can use to change the world.", "author": "Nelson Mandela", "emoji": "💪"},
    ],
    "Neutral": [
        {"text": "Bugün öğrendiğin her şey, yarınki sen için bir armağandır.", "author": "Benjamin Franklin", "emoji": "💭"},
        {"text": "The secret of getting ahead is getting started.", "author": "Mark Twain", "emoji": "🎯"},
        {"text": "Küçük adımlar, büyük yolculuklar yapar.", "author": "Lao Tzu", "emoji": "🪐"},
        {"text": "An investment in knowledge pays the best interest.", "author": "Benjamin Franklin", "emoji": "💡"},
        {"text": "Merak, bilgeliğin anasıdır. Sormaya devam et.", "author": "Albert Einstein", "emoji": "🔭"},
        {"text": "You don't have to be great to start, but you have to start to be great.", "author": "Zig Ziglar", "emoji": "🌱"},
        {"text": "Zihin, doldurulacak bir kova değil, yakılacak bir ateştir.", "author": "Plutarch", "emoji": "🔥"},
        {"text": "Live as if you were to die tomorrow. Learn as if you were to live forever.", "author": "Mahatma Gandhi", "emoji": "♾️"},
    ]
}


# ─── MockLLM ───────────────────────────────────────────────────────────────────
class MockLLM:
    def __init__(self, agent_type="PLAN"):
        self.agent_type = agent_type

    def invoke(self, messages):
        user_msg = messages[-1].content if messages else ""
        lang = "tr"

        responses = {
            "tr": {
                "GOAL": [
                    "# Hedef Belirleme\nHarika bir hedef! 🎯 Bu hedefe ulaşmak için hangi adımları atmalıyız? Beraber planlayalım.",
                    "# Vizyon ve Hedef\nHedefin net ve ilham verici. 🌟 Bunu başarılabilir küçük parçalara bölelim!",
                ],
                "PLAN": [
                    "# Haftalık Ders Programı\nSenin için taslak bir plan: \n1. Konu Tekrarı (45 dk)\n2. Soru Çözümü (30 dk)\n3. Dinlenme (15 dk)\nHaydi başlayalım! 📅",
                    "# Verimli Çalışma Stratejisi\nÖnerim: Zor konuları sabah erkende, pratiği öğleden sonra yap. ⏳ Verimlilik artacak!",
                ],
                "FEEDBACK": [
                    "# Performans Analizi\nHer sınav bir öğrenme fırsatıdır. 📈 Nerede zorlandığını beraber inceleyelim.",
                    "# Gelişim Raporu\nBu sonuç gelişiminin bir parçası. 🎓 Zayıf olduğun noktaları güçlendirelim.",
                ],
                "HEALTH": [
                    "# Sağlık ve Odaklanma\nBugün yeterince su içtin mi? 💧 Beyin en iyi yakıtla çalışır.",
                    "# Uyku ve Dinlenme\nİyi bir gece uykusu 10 saatlik çalışmadan daha etkilidir. 😴 Uyku düzenine bak!",
                ],
                "TEACH": [
                    "# Konu Özeti\nBu konuyu senin için detaylıca özetledim. 📚 Temel noktaları kavraman çok önemli.",
                    "# Uygulama Testi\nBilgini ölçmek için harika bir test hazırladım! 📝 Hadi soruları cevapla.",
                    "# Pratik Sorular\nKonuyu pekiştirmen için sana özel sorular hazırladım. 🤔 Başarılar dilerim!",
                ],
            },
            "en": {
                "GOAL": [
                    "Great goal! 🎯 What steps do we need to take to achieve it? Let's plan together.",
                    "Your goal is clear and inspiring. 🌟 Let's break it into small achievable steps!",
                ],
                "PLAN": [
                    "Here's a draft plan for you: \n1. Topic Review (45 min)\n2. Practice (30 min)\n3. Break (15 min)\nLet's go! 📅",
                    "Suggestion: Tackle hard topics in the morning, practice in the afternoon. ⏳",
                ],
                "FEEDBACK": [
                    "Every exam is a learning opportunity. 📈 Let's see where you can improve.",
                    "This result is part of your growth. 🎓 Let's strengthen your weak points.",
                ],
                "HEALTH": [
                    "Did you drink enough water today? 💧 Your brain runs best when hydrated.",
                    "A good night's sleep beats 10 hours of tired studying. 😴 Check your sleep schedule!",
                ],
                "TEACH": [
                    "Let me explain this step by step. 📚 Starting from the basics...",
                    "Great question! 🤔 Here are the most important things to know about this topic...",
                ],
            }
        }

        # detect language roughly
        tr_chars = set("çğışöüÇĞİŞÖÜ")
        if any(ch in tr_chars for ch in user_msg):
            lang = "tr"
        elif any(w in user_msg.lower() for w in ["hello", "help me", "i want", "study", "create", "make"]):
            lang = "en"

        lib = responses.get(lang, responses["tr"])
        return AIMessage(content=random.choice(lib.get(self.agent_type, lib.get("GOAL", ["Sana yardımcı olmak için buradayım! 💙"]))))


# ─── LLM Getter ────────────────────────────────────────────────────────────────
def get_llm(agent_type="PLAN"):
    if os.getenv("ANTHROPIC_API_KEY"):
        print(f"DEBUG: Using Anthropic for {agent_type}", flush=True)
        return ChatAnthropic(model="claude-sonnet-4-20250514")
    elif os.getenv("GROQ_API_KEY"):
        print(f"DEBUG: Using Groq for {agent_type}", flush=True)
        return ChatGroq(model="llama-3.3-70b-versatile")
    elif os.getenv("GOOGLE_API_KEY"):
        print(f"DEBUG: Using Google for {agent_type}", flush=True)
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    else:
        print(f"DEBUG: Using MockLLM for {agent_type}", flush=True)
        return MockLLM(agent_type)


# ─── Generic Node ──────────────────────────────────────────────────────────────
def generic_node(state: AgentState, agent_type: str, system_prompts: dict):
    lang = state.get("language", "tr")
    llm = get_llm(agent_type)

    system_msg = system_prompts.get(lang, system_prompts.get("en", ""))
    user_name = state.get("user_name", "") or ("Öğrenci" if lang == "tr" else "Student")
    system_msg = system_msg.replace("[name]", user_name)
    # Also inject language instruction so LLM can't drift
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
        if not isinstance(llm, MockLLM):
            fallback_llm = MockLLM(agent_type)
            response = fallback_llm.invoke(messages)
        else:
            raise e

    return {"messages": [response]}


# ─── Specialist Nodes ──────────────────────────────────────────────────────────
def goal_node(state: AgentState):
    prompts = {
        "tr": (
            "Sen bir hedef belirleme koçusun. SADECE hedef belirleme konusunda yardım et. "
            "Kesinlikle ders anlatma veya ders programı hazırlama. "
            "[name] için gerçekçi, ölçülebilir hedefler belirle. Sıcak ve destekleyici ol."
        ),
        "en": (
            "You are a goal-setting coach. ONLY help with setting goals. "
            "DO NOT explain topics or create study schedules. "
            "Help [name] set realistic, measurable goals. Be warm and supportive."
        ),
    }
    return generic_node(state, "GOAL", prompts)


def teach_node(state: AgentState):
    user_msg = state["messages"][-1].content.lower()
    is_test = any(w in user_msg for w in ["test", "quiz", "soru", "sor", "sınav yap", "question", "ask me"])
    
    prompts = {
        "tr": (
            f"Sen sabırlı bir öğretmensin. [name] için {'konuyu test et ve sorular sor' if is_test else 'konuyu açıkla ve özet çıkar'}. "
            "SADECE eğitim içeriği ver. Kişisel motivasyon veya planlama yapma. "
            "Anlaşılır, adım adım ve eğitici bir şekilde anlat. "
            f"Yanıtına MUTLAKA '# [Konu Başlığı]' veya '# [Test Adı]' şeklinde bir başlıkla başla. "
            f"{'Öğrenci test olmak istiyor, ona 3-5 tane çoktan seçmeli veya açık uçlu soru sor.' if is_test else 'Öğrenci konuyu öğrenmek istiyor, ona detaylı ve madde madde bir özet ver.'}"
        ),
        "en": (
            f"You are a patient teacher. {'Test the student and ask questions' if is_test else 'Explain the topic and create a summary'} for [name]. "
            "ONLY provide educational content. Do not give personal motivation or planning. "
            "Be clear, step-by-step, and educational. "
            f"ALWAYS start your response with a '# [Topic Title]' or '# [Test Name]' header. "
            f"{'The student wants to be tested, ask 3-5 multiple choice or open-ended questions.' if is_test else 'The student wants to learn the topic, provide a detailed bullet-point summary.'}"
        ),
    }
    return generic_node(state, "TEACH", prompts)


def plan_node(state: AgentState):
    prompts = {
        "tr": (
            "Sen bir eğitim planlama uzmanısın. SADECE verimli çalışma programları ve zaman yönetimi stratejileri hazırla. "
            "Çizelgeleri Markdown tablosu formatında sun (Sütunlar: Gün | Saat | Konu | Aktivite). "
            "Konu anlatma veya motivasyon verme. Pratik ve uygulanabilir ol."
        ),
        "en": (
            "You are an education planning expert. ONLY prepare efficient study schedules and time management strategies. "
            "Present schedules in Markdown table format (Columns: Day | Time | Subject | Activity). "
            "Do NOT explain topics or give motivation. Be practical and actionable."
        ),
    }
    return generic_node(state, "PLAN", prompts)


def feedback_node(state: AgentState):
    prompts = {
        "tr": (
            "Sen bir geri bildirim ve gelişim koçusun. SADECE sınav sonuçlarını veya performansı analiz et. "
            "Hatalardan ders çıkarmayı teşvik et. Kısa ve net ol. Yeni ders programı hazırlama."
        ),
        "en": (
            "You are a feedback and growth coach. ONLY analyze exam results or performance. "
            "Encourage learning from mistakes. Be concise and clear. Do NOT create a new study schedule."
        ),
    }
    return generic_node(state, "FEEDBACK", prompts)


def health_node(state: AgentState):
    prompts = {
        "tr": (
            "Sen bir sağlık ve wellness rehberisin. [name]'e SADECE uyku, beslenme, spor ve stres yönetimi "
            "konularında tavsiyeler ver. Akademik tavsiye verme. Sıcak ve samimi ol."
        ),
        "en": (
            "You are a health and wellness guide. Give [name] advice ONLY on sleep, nutrition, exercise, "
            "and stress management. Do NOT give academic advice. Be warm and genuine."
        ),
    }
    return generic_node(state, "HEALTH", prompts)


def motivation_node(state: AgentState):
    lang = state.get("language", "tr")
    sentiment = state.get("sentiment", "Neutral")
    user_name = state.get("user_name", "") or ("Arkadaşım" if lang == "tr" else "Friend")

    # Select quote matching sentiment
    quote_list = QUOTES.get(sentiment, QUOTES["Neutral"])
    quote = random.choice(quote_list)

    if lang == "tr":
        templates = [
            f"Hey {user_name} {quote['emoji']} Bugün biraz ağır hissettirdi değil mi? {quote['author']} şöyle demiş: \"{quote['text']}\" — Sen bu yolu yürüyebilirsin, sana inanıyorum! 💙",
            f"Selam {user_name}! ✨ Enerjinin nasıl olduğunu hissedebiliyorum. {quote['author']}: \"{quote['text']}\" {quote['emoji']} Derin bir nefes al ve yeniden başla — bu sefer daha da güçlüsün.",
            f"Canım {user_name}, bazen durup dinlenmek gerekir. 🌿 {quote['author']}'un bu sözü tam senin için: \"{quote['text']}\" — Sen en iyisini hak ediyorsun. Devam et! 🔥",
        ]
    else:
        templates = [
            f"Hey {user_name} {quote['emoji']} Sounds like today was a bit heavy. Here's what {quote['author']} had to say: \"{quote['text']}\" — You've absolutely got this! 💙",
            f"Hi {user_name}! ✨ I can sense where you're at. {quote['author']} once said: \"{quote['text']}\" {quote['emoji']} Take a deep breath — you're stronger than you think.",
            f"Dear {user_name}, sometimes we just need a moment. 🌿 This is for you, from {quote['author']}: \"{quote['text']}\" — Keep going, you deserve the best! 🔥",
        ]

    response_text = random.choice(templates)

    # Try to enhance with LLM
    llm = get_llm("MOTIVATION")
    if not isinstance(llm, MockLLM):
        try:
            lang_instruction = (
                "MUTLAKA Türkçe yanıt ver. İngilizce kullanma." if lang == "tr"
                else "ALWAYS reply in English. Do not use Turkish."
            )
            system_prompt = (
                f"{lang_instruction}\n"
                f"You are a warm, caring older sibling who is also an AI coach. "
                f"The user's name is {user_name}. Their emotional state is: {sentiment}. "
                f"Use this philosopher quote as the heart of your message: '{quote['text']}' by {quote['author']}. "
                f"Be personal, warm, never robotic. Address them by name. "
                f"End with a relevant emoji and one personal encouragement sentence. "
                f"Keep response to 3-4 sentences max."
            )
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = llm.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            print(f"Motivation LLM Error: {e}")

    return {"messages": [AIMessage(content=response_text)]}
