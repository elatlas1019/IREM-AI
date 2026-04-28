# 💠 IREM AI - Smart Educational Coach / Akıllı Eğitim Koçu

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-v0.100%2B-009688.svg)

**IREM AI** is a professional, multi-agent artificial intelligence system designed to support students in their academic journey. It provides personalized study plans, topic explanations, interactive tests, and emotional coaching through a stunning, modern web interface.

**IREM AI**, öğrencilerin akademik yolculuklarını desteklemek için tasarlanmış profesyonel, çok ajanlı bir yapay zeka sistemidir. Modern ve etkileyici bir web arayüzü üzerinden kişiselleştirilmiş ders planları, konu anlatımları, interaktif testler ve duygusal koçluk sunar.

---

## 🌟 Key Features / Temel Özellikler

- **🤖 Multi-Agent Orchestration**: Specialized agents for goal setting, study planning, teaching, and feedback.
- **📈 Dynamic Progress Tracking**: Real-time analytics, study goals, and a timed academic calendar.
- **💡 Smart Study Content**: Intelligent generation of topic summaries and tests with PDF export capability.
- **🎙️ Voice Integration**: Hands-free interaction via Web Speech API.
- **🔔 Smart Reminders**: Automated notifications for scheduled study sessions with sound alerts.
- **🎨 Premium UI/UX**: Dark mode by default, glassmorphism effects, and smooth animations.

---

## 🛠️ Tech Stack / Teknoloji Yığını

- **Backend**: Python, FastAPI, LangGraph, LangChain, SQLite.
- **AI Models**: Support for Groq (Llama 3), Google (Gemini), and Anthropic (Claude).
- **Frontend**: HTML5, Vanilla CSS3 (Custom Design System), JavaScript (ES6+).
- **Export**: ReportLab for high-quality PDF generation with Turkish character support.

---

## 🚀 Quick Start / Hızlı Başlangıç

### 1. Prerequisites / Gereksinimler
- Python 3.10+
- At least one API Key (Groq, Google, or Anthropic)

### 2. Installation / Kurulum
```bash
# Clone the repository
git clone https://github.com/yourusername/IREM-AI.git
cd IREM-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration / Yapılandırma
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
# Optional
ANTHROPIC_API_KEY=your_key_here
```

### 4. Run / Çalıştır
```bash
cd backend
python -m uvicorn main:app --reload
```
Open `http://localhost:8000/dashboard` in your browser.

---

## 🌍 Localization / Yerelleştirme
Fully bilingual support (Turkish & English). The system automatically detects user intent and switches context while maintaining high-quality responses in both languages.

Tam çift dilli destek (Türkçe ve İngilizce). Sistem, kullanıcı amacını otomatik olarak algılar ve her iki dilde de yüksek kaliteli yanıtlar sunarken bağlamı korur.

---

## 📜 License / Lisans
This project is licensed under the MIT License.

---

Developed with ❤️ for the future of education.  
Eğitimin geleceği için ❤️ ile geliştirildi.
