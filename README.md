[README (1).md](https://github.com/user-attachments/files/27233205/README.1.md)

<div align="center">

# ✦ IREM AI
### Kişisel Öğrenme Koçun — Your Personal Learning Coach

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-irem--ai.streamlit.app-7C3AED?style=for-the-badge)](https://irem-ai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-3B82F6?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

</div>

---

## 🇹🇷 Türkçe

**IREM AI**, öğrencilere yönelik geliştirilmiş yapay zeka destekli kişisel bir öğrenme koçudur. LangGraph ile yönetilen çok ajanlı bir mimari üzerine kurulu olan sistem; Streamlit arayüzü üzerinden konu anlatımı, ders planı oluşturma ve çalışma takibini tek bir ekranda sunar.

Ama IREM AI bundan çok daha fazlası.

> **Sadece "ne öğrenmeliyim?" sorusunu değil, "bu an nasıl hissediyorum?" sorusunu da anlayan bir sistemdir.**

Kullanıcı akademik bir soru sorduğunda da, *"heyecanlıyım"*, *"kararsızım"*, *"çok yorgunum"*, *"aşık oldum"*, *"mutsuzum"* gibi duygusal bir şey yazdığında da — IREM AI dinler, anlar ve çalışma alanından kişiselleştirilmiş bir destek, motivasyon veya yönlendirme sunar. Psikolojik farkındalığa sahip bir mentör gibi davranır.

---

### 🎯 Ne Yapar?

**📚 Akademik Destek**
- Konu anlatımı, özet, soru çözümü: Kullanıcının yazdığı içerik AI ajanları tarafından işlenerek hem sohbet penceresine hem de ayrı bir çalışma alanı paneline aktarılır.
- Ders planı ve çalışma programı önerileri.

**🧠 Duygusal Destek & Mentörlük**
- "Sınavdan çok korkuyorum", "Motivasyonum sıfır", "Bugün hiçbir şey yapamıyorum" gibi ifadelere akademik içerik değil; empati, psikolojik destek ve somut yönlendirme ile karşılık verir.
- Kullanıcının ruh hali ve enerji durumu sisteme yansır; yanıtlar buna göre şekillenir.
- Uzman bir mentörün yapacağı gibi: önce anlar, sonra yönlendirir.

**✨ Günün Sözü ile Güne Başlamak**
- Her gün, uygulama açıldığında ekranda yeni bir motivasyon sözü belirir.
- Küçük ama kasıtlı bir tasarım kararı: çalışmaya başlamadan önce zihni hazırlamak.

**🎙️ Sesli Giriş**
- Groq Whisper API ile Türkçe ses tanıma. Mikrofon butonuna tıkla, konuş, sistem metne çevirir.

**🎯 Hedef Takibi**
- Tarih aralıklı hedefler eklenebilir, ilerleme çubukları ile görsel takip. SQLite ile kalıcı depolama.

**📅 Çalışma Takvimi**
- Gün ve saat bazlı çalışma oturumları planlanır. Aktif oturum başladığında ekranda canlı uyarı görünür.

**⏰ Otomatik Hatırlatıcı**
- 60 saniyede bir arkaplanda aktif çalışma oturumu kontrol edilir, toast bildirimi tetiklenir.

**📥 Not İndirme**
- Çalışma alanındaki içerik `.txt` formatında indirilebilir.

---

### 🔍 Benzer Uygulamalardan Farkı

| Özellik | Genel AI Chatbot'lar | Diğer Eğitim AI'ları | IREM AI |
|---|---|---|---|
| Akademik konu anlatımı | ✓ | ✓ | ✓ |
| **Duygusal destek & mentörlük** | ✗ | ✗ | ✓ Psikolojik farkındalıklı yanıt |
| **Günlük motivasyon sözü** | ✗ | ✗ | ✓ Her gün yenilenir |
| Çalışma alanı paneli | ✗ | ✗ | ✓ Ayrı içerik bölmesi |
| Türkçe sesli giriş | ✗ | ✗ | ✓ Groq Whisper (tr) |
| Hedef + takvim modülü | ✗ | kısmen | ✓ Entegre, görsel |
| Aktif oturum hatırlatıcısı | ✗ | ✗ | ✓ 60s otomatik kontrol |
| Multi-agent orkestrasyon | ✗ | ✗ | ✓ LangGraph graph |

---

### 🛠️ Mimari

```
Kullanıcı (Streamlit UI)
    │
    ├─ Metin / Ses Girişi
    │       └─ Groq Whisper (STT) ──► Türkçe metin
    │
    ├─ LangGraph Coaching Agent
    │       ├─ Planlama Ajanı         ← akademik hedefler
    │       ├─ Öğretme Ajanı          ← konu anlatımı, soru çözümü
    │       ├─ Duygusal Destek Ajanı  ← his/motivasyon tespiti → yönlendirme
    │       └─ Geri Bildirim Ajanı
    │               └─ LLM: Groq (Llama 3) / Google Gemini / Anthropic Claude
    │
    ├─ Çalışma Alanı Paneli  ◄── Agent çıktısı (akademik veya duygusal)
    ├─ Günün Sözü            ◄── Her açılışta rastgele, motivasyon odaklı
    ├─ Hedef Takibi          ◄── SQLite
    └─ Çalışma Takvimi       ◄── Session state + 60s autorefresh
```

---

### 🚀 Kurulum

```bash
git clone https://github.com/elatlas1019/IREM-AI.git
cd IREM-AI

python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate

pip install -r requirements.txt
```

`.env` dosyası oluştur:

```env
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here   # opsiyonel
```

Çalıştır:

```bash
streamlit run app.py
```

---

## 🇬🇧 English

**IREM AI** is an AI-powered personal learning coach built for students. It runs on a multi-agent architecture orchestrated by LangGraph, and delivers topic explanations, study planning, and progress tracking — all within a single Streamlit interface.

But IREM AI is much more than that.

> **It doesn't just answer "what should I study?" — it also understands "how am I feeling right now?"**

When a user types an academic question, IREM AI produces structured study content. When a user types *"I'm so tired"*, *"I feel lost"*, *"I just fell in love and can't focus"*, or *"I'm anxious about my exam"* — IREM AI responds not with a study plan, but with empathy, psychological grounding, and actionable guidance. It behaves like a mentor who listens first, then leads.

---

### 🎯 What It Does

**📚 Academic Support**
- Topic explanations, summaries, problem solving: output appears in both the chat and a dedicated workspace panel side by side.
- Personalized study planning based on goals and schedule.

**🧠 Emotional Support & Mentorship**
- Detects emotional states from natural language ("I'm overwhelmed", "I have no motivation", "I can't do anything today") and responds with empathy-first, psychologically aware guidance.
- The user's mood and energy level are tracked and influence how the system responds throughout the session.
- Not a replacement for professional support — but a first, always-available layer of mentorship.

**✨ Daily Motivational Quote**
- Every time the app loads, a fresh motivational quote appears on screen.
- A small but intentional design choice: prime the mind before the work begins.

**🎙️ Voice Input**
- Turkish speech-to-text via Groq Whisper API. Click the mic, speak, the system transcribes automatically.

**🎯 Goal Tracking**
- Add date-range goals and track progress visually with dynamic progress bars. Persisted in SQLite.

**📅 Study Calendar**
- Plan sessions by day and time. Active sessions trigger live alerts in the sidebar.

**⏰ Auto Reminders**
- Background check every 60 seconds — toast notification fires when an active session starts.

**📥 Note Export**
- Workspace content downloadable as `.txt`.

---

### 🔍 What Makes It Different

Most AI tools are either academic tutors or general chatbots. IREM AI is neither — it's a context-aware coach that responds to the full picture of a student's day: what they need to learn *and* how they feel while trying to learn it.

The combination of emotional intelligence, daily motivation, structured goal tracking, and multi-agent academic support in a single interface is what sets it apart.

---

### 🛠️ Architecture

```
User (Streamlit UI)
    │
    ├─ Text / Voice Input
    │       └─ Groq Whisper (STT) ──► Turkish transcription
    │
    ├─ LangGraph Coaching Agent
    │       ├─ Planning Agent          ← academic goals
    │       ├─ Teaching Agent          ← explanations, problem solving
    │       ├─ Emotional Support Agent ← mood detection → empathetic guidance
    │       └─ Feedback Agent
    │               └─ LLM Backend: Groq (Llama 3) / Google Gemini / Anthropic Claude
    │
    ├─ Workspace Panel   ◄── Agent output (academic or emotional)
    ├─ Daily Quote       ◄── Refreshed each session, motivation-first
    ├─ Goal Tracker      ◄── SQLite persistence
    └─ Study Calendar    ◄── Session state + 60s autorefresh
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit, Custom CSS (dark glassmorphism) |
| Agent Orchestration | LangGraph, LangChain |
| LLM Backends | Groq (Llama 3.3 70B), Google Gemini, Anthropic Claude |
| Speech-to-Text | Groq Whisper (`whisper-large-v3`, Turkish) |
| Database | SQLite (goals persistence) |
| Deployment | Streamlit Cloud |

---

## License

MIT License — free to use, modify, and distribute.

---

<div align="center">

---

**Designed & Developed by**

### İrem Burcu Orhan

*Eğitim teknolojileri ve yapay zeka alanında, öğrencinin hem zihnini hem kalbini anlayan bir sistem inşa etme vizyonuyla.*
*With the vision of building a system that understands not just the student's mind, but also their heart.*

[![GitHub](https://img.shields.io/badge/GitHub-elatlas1019-181717?style=flat-square&logo=github)](https://github.com/elatlas1019)

---

  <sub>Built for the student who needs both a teacher and someone who listens.<br>Hem bir öğretmene hem de dinleyecek birine ihtiyaç duyan öğrenci için yapıldı.</sub>
</div>
