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
if "workspace_content" not in st.session_state:
    st.session_state.workspace_content = ""
if "goals_list" not in st.session_state:
    st.session_state.goals_list = []
if "schedule_list" not in st.session_state:
    st.session_state.schedule_list = []

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
            import base64
            # Convert audio to base64
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            st.info("Sesiniz işleniyor... (Multimodal STT)")
            # For now, we simulate the transcription or use a placeholder
            prompt = "Sesli komut (simüle edilmiş): Bana bugünkü ders programımı anlat."
            # In a real scenario, we would send audio_b64 to an LLM/STT API here.

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            config = {"configurable": {"thread_id": "st-session-1"}}
            state_update = {"messages": [HumanMessage(content=prompt)], "user_name": st.session_state.user_name, "language": "tr"}
            
            try:
                # Use current event loop since nest_asyncio is applied
                response_state = asyncio.run(coaching_agent_app.ainvoke(state_update, config))
                
                last_msg = response_state["messages"][-1].content
                st.session_state.energy = response_state.get("energy_score", st.session_state.energy)
                st.session_state.messages.append({"role": "assistant", "content": last_msg})
                
                # Update Workspace Content
                st.session_state.workspace_content = last_msg
            except Exception as e:
                st.error(f"Hata: {e}")
            
            st.rerun()

    with col_doc:
        st.markdown("### 📄 Çalışma Alanı")
        if st.session_state.workspace_content:
            content = st.session_state.workspace_content
            with st.container(height=400, border=True):
                st.markdown(content)
            st.download_button("📥 Notu İndir", data=content.encode('utf-8'), file_name="irem_not.txt", mime="text/plain; charset=utf-8", use_container_width=True)
        else:
            st.info("Notların burada görünecek.")

elif st.session_state.current_panel == "goals":
    st.markdown("## 🎯 Hedeflerim")
    col_f, col_v = st.columns([1, 1.5])
    
    with col_f:
        with st.form("goal_form_modern"):
            g_title = st.text_input("Hedef Adı")
            g_desc = st.text_area("Açıklama")
            g_start = st.date_input("Başlangıç Tarihi", datetime.now())
            g_end = st.date_input("Bitiş Tarihi", datetime.now())
            if st.form_submit_button("Hedef Ekle"):
                new_goal = {"title": g_title, "desc": g_desc, "start": str(g_start), "end": str(g_end), "status": "Devam Ediyor"}
                st.session_state.goals_list.append(new_goal)
                
                # Persist to SQLite
                conn = sqlite3.connect('goals.db')
                c = conn.cursor()
                c.execute("CREATE TABLE IF NOT EXISTS goals_v2 (title TEXT, description TEXT, start_date TEXT, end_date TEXT, status TEXT)")
                c.execute("INSERT INTO goals_v2 VALUES (?, ?, ?, ?, ?)", (g_title, g_desc, str(g_start), str(g_end), "Devam Ediyor"))
                conn.commit(); conn.close()
                st.success("Hedef başarıyla eklendi!")

    with col_v:
        st.markdown("### 📅 Aylık Hedef Takvimi")
        if st.session_state.goals_list:
            import plotly.express as px
            df_goals = pd.DataFrame(st.session_state.goals_list)
            df_goals['start'] = pd.to_datetime(df_goals['start'])
            df_goals['end'] = pd.to_datetime(df_goals['end'])
            
            fig = px.timeline(df_goals, x_start="start", x_end="end", y="title", color="status", 
                             title="Hedef Zaman Çizelgesi", template="plotly_dark")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Henüz bir hedef eklemediniz.")
    
    st.markdown("---")
    st.markdown("### Tüm Hedefler")
    if st.session_state.goals_list:
        st.table(st.session_state.goals_list)

elif st.session_state.current_panel == "calendar":
    st.markdown("## 📅 Çalışma Takvimi")
    
    # Reminder logic
    now = datetime.now()
    for session in st.session_state.schedule_list:
        s_date = datetime.strptime(session["day"], "%Y-%m-%d").date()
        if s_date == now.date():
            s_time = datetime.strptime(session["start_time"], "%H:%M:%S").time()
            # Simple reminder: if session starts in the next 30 minutes
            if 0 <= (datetime.combine(now.date(), s_time) - now).total_seconds() <= 1800:
                st.toast(f"🔔 Yaklaşan Oturum: {session['task']} (Saat {session['start_time']})", icon="📅")

    col_f, col_v = st.columns([1, 2])
    
    with col_f:
        with st.form("cal_form_modern"):
            c_task = st.text_input("Görev/Ders")
            c_day = st.date_input("Gün", datetime.now())
            c_start_t = st.time_input("Başlangıç Saati", value=now.time())
            c_end_t = st.time_input("Bitiş Saati", value=now.time())
            if st.form_submit_button("Takvime Ekle"):
                new_session = {
                    "task": c_task, 
                    "day": str(c_day), 
                    "start_time": str(c_start_t), 
                    "end_time": str(c_end_t),
                    "start": f"{c_day} {c_start_t}",
                    "end": f"{c_day} {c_end_t}"
                }
                st.session_state.schedule_list.append(new_session)
                st.success(f"{c_task} kaydedildi!")

    with col_v:
        st.markdown("### 🗓️ Haftalık Görünüm")
        if st.session_state.schedule_list:
            import plotly.express as px
            df_sched = pd.DataFrame(st.session_state.schedule_list)
            df_sched['start'] = pd.to_datetime(df_sched['start'])
            df_sched['end'] = pd.to_datetime(df_sched['end'])
            
            fig = px.timeline(df_sched, x_start="start", x_end="end", y="task", color="task",
                             title="Haftalık Çalışma Planı", template="plotly_dark")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Henüz bir çalışma oturumu eklemediniz.")
