import streamlit as st
import os
import asyncio
import sqlite3
import random
import pandas as pd
from datetime import datetime
from audio_recorder_streamlit import audio_recorder
from langchain_core.messages import HumanMessage
from backend.main import create_pdf_buffer
from backend.agents.graph import coaching_agent_app
from backend.agents.specialized import QUOTES

# --- PAGE CONFIG ---
st.set_page_config(page_title="IREM AI - Eğitim Koçu", layout="wide", page_icon="💠")

# --- CUSTOM CSS (Modern Design with Cards) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Outfit', sans-serif; }
    
    .stApp {
        background: #0F172A;
        color: #F8FAFC;
    }
    
    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid rgba(124, 58, 237, 0.2);
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .chat-bubble {
        padding: 15px 20px;
        border-radius: 20px;
        margin-bottom: 15px;
        max-width: 85%;
    }
    
    .user-msg {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
        color: white;
        margin-left: auto;
    }
    
    .assistant-msg {
        background: #1E293B;
        color: #F1F5F9;
        margin-right: auto;
        border-left: 4px solid #7C3AED;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #7C3AED 0%, #EC4899 100%);
        color: white;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = "Öğrenci"
if "grade" not in st.session_state:
    st.session_state.grade = "12. Sınıf"
if "mood" not in st.session_state:
    st.session_state.mood = "Pozitif"
if "energy" not in st.session_state:
    st.session_state.energy = 85
if "current_panel" not in st.session_state:
    st.session_state.current_panel = "dashboard"

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>💠 IREM AI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.session_state.user_name = st.text_input("Adın Soyadın", value=st.session_state.user_name)
    st.session_state.grade = st.selectbox("Sınıfın", ["9. Sınıf", "10. Sınıf", "11. Sınıf", "12. Sınıf", "Mezun"], index=3)
    
    st.markdown("### Navigasyon")
    if st.sidebar.button("📊 Dashboard", use_container_width=True, key="nav_dash"):
        st.session_state.current_panel = "dashboard"
        st.rerun()
    if st.sidebar.button("🎯 Hedeflerim", use_container_width=True, key="nav_goals"):
        st.session_state.current_panel = "goals"
        st.rerun()
    if st.sidebar.button("📅 Çalışma Takvimi", use_container_width=True, key="nav_cal"):
        st.session_state.current_panel = "calendar"
        st.rerun()
    
    st.markdown("---")
    st.progress(st.session_state.energy / 100, text=f"Enerji: %{st.session_state.energy}")

# --- MAIN DASHBOARD ---
if st.session_state.current_panel == "dashboard":
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f'<div class="metric-card"><div style="color:#94A3B8;">ODAK</div><div style="font-size:1.8rem; font-weight:700; color:#7C3AED;">%{st.session_state.energy}</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><div style="color:#94A3B8;">MOD</div><div style="font-size:1.8rem; font-weight:700; color:#EC4899;">{st.session_state.mood}</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><div style="color:#94A3B8;">GÜN</div><div style="font-size:1.8rem; font-weight:700; color:#3B82F6;">{datetime.now().strftime("%d %b")}</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="metric-card"><div style="color:#94A3B8;">SAAT</div><div style="font-size:1.8rem; font-weight:700; color:#10B981;">{datetime.now().strftime("%H:%M")}</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Quote of the day (Robust Fix)
    all_quotes = QUOTES.get("Motivation", [{"text": "Başarı, çabanın sonucudur.", "author": "İrem AI", "emoji": "✨"}])
    quote = random.choice(all_quotes)
    st.markdown(f"""
        <div style="background: rgba(124, 58, 237, 0.1); padding: 25px; border-radius: 20px; border: 1px solid rgba(124, 58, 237, 0.3); text-align: center;">
            <div style="font-size:11px; color:#F472B6; font-weight:800; letter-spacing:2px; margin-bottom:5px;">✨ GÜNÜN SÖZÜ</div>
            <div style="font-size:1.1rem; font-weight:600; color:white;">"{quote['text']}"</div>
            <div style="font-size:0.9rem; color:#F472B6; margin-top:5px;">— {quote['author']} {quote['emoji']}</div>
        </div>
    """, unsafe_allow_html=True)

    col_chat, col_doc = st.columns([1.1, 1.4], gap="large")
    
    with col_chat:
        chat_sub = st.container(height=450, border=False)
        with chat_sub:
            if not st.session_state.messages:
                st.markdown(f'<div class="chat-bubble assistant-msg">Merhaba {st.session_state.user_name}! Bugün sana nasıl yardımcı olabilirim? 💠</div>', unsafe_allow_html=True)
            for m in st.session_state.messages:
                cls = "user-msg" if m["role"] == "user" else "assistant-msg"
                st.markdown(f'<div class="chat-bubble {cls}">{m["content"]}</div>', unsafe_allow_html=True)
        
        # Audio
        audio_bytes = audio_recorder(text="🎙️", icon_size="2x", key="recorder_dash")
        prompt = st.chat_input("Mesajınızı yazın...")
        
        if audio_bytes and len(audio_bytes) > 0:
            st.info("Sesli komut şu an devre dışıdır.")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            config = {"configurable": {"thread_id": "st-session-1"}}
            state_update = {"messages": [HumanMessage(content=prompt)], "user_name": st.session_state.user_name, "language": "tr"}
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_state = loop.run_until_complete(coaching_agent_app.ainvoke(state_update, config))
            
            last_msg = response_state["messages"][-1].content
            st.session_state.energy = response_state.get("energy_score", st.session_state.energy)
            st.session_state.messages.append({"role": "assistant", "content": last_msg})
            st.rerun()

    with col_doc:
        st.markdown("### 📄 Çalışma Alanı")
        last_assist = next((m for m in reversed(st.session_state.messages) if m.get("role") == "assistant"), None)
        if last_assist:
            content = last_assist["content"]
            with st.container(height=400, border=True):
                st.markdown(content)
            st.download_button("📥 Notu İndir", data=content.encode('utf-8'), file_name="irem_not.txt", mime="text/plain; charset=utf-8", use_container_width=True)
        else:
            st.info("Notların burada görünecek.")

elif st.session_state.current_panel == "goals":
    st.markdown("## 🎯 Hedeflerim")
    with st.form("goal_form_modern"):
        g_title = st.text_input("Hedef")
        g_desc = st.text_area("Açıklama")
        g_date = st.date_input("Hedef Tarihi")
        if st.form_submit_button("Hedef Ekle"):
            conn = sqlite3.connect('goals.db')
            c = conn.cursor()
            c.execute("CREATE TABLE IF NOT EXISTS goals (title TEXT, description TEXT, date TEXT)")
            c.execute("INSERT INTO goals VALUES (?, ?, ?)", (g_title, g_desc, str(g_date)))
            conn.commit(); conn.close()
            st.success("Hedef eklendi!")
    
    try:
        conn = sqlite3.connect('goals.db')
        df = pd.read_sql_query("SELECT * FROM goals", conn)
        conn.close()
        st.dataframe(df, use_container_width=True)
    except: st.info("Henüz hedef yok.")

elif st.session_state.current_panel == "calendar":
    st.markdown("## 📅 Çalışma Takvimi")
    with st.form("cal_form_modern"):
        c_task = st.text_input("Görev/Ders")
        c_day = st.date_input("Gün")
        c_time = st.time_input("Saat")
        if st.form_submit_button("Takvime Ekle"):
            st.success(f"{c_task} görevi {c_day} saat {c_time} için kaydedildi!")
