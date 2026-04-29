import streamlit as st
import os
import json
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.agents.graph import coaching_agent_app
from backend.main import create_pdf_buffer
from backend.database import SessionLocal, engine
from backend import crud, models, schemas
from backend.agents.specialized import QUOTES
import random
import sqlite3
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
import plotly.express as px
import pandas as pd

# Initialize database
models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

# --- LOCAL DB INITIALIZATION (Fix 1) ---
def init_local_db():
    conn = sqlite3.connect('goals.db')
    c = conn.cursor()
    # Goals table
    c.execute('''CREATE TABLE IF NOT EXISTS goals 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, description TEXT, 
                  start_date TEXT, end_date TEXT, status TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    # Calendar events table (Fix 2)
    c.execute('''CREATE TABLE IF NOT EXISTS events 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, date TEXT, 
                  start_time TEXT, end_time TEXT, duration INTEGER)''')
    conn.commit()
    conn.close()

init_local_db()

# OpenAI Client (Fix 4)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load env
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="IREM AI - Smart Educational Coach",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME COLORS & CSS (Directly from index.html) ---
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: #0F172A !important;
        font-family: 'Inter', sans-serif !important;
        color: #F1F5F9 !important;
    }
    
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid rgba(124, 58, 237, 0.2);
        width: 200px !important;
    }
    
    /* Sidebar Components */
    .logo-text {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7C3AED, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        display: block;
    }
    
    .nav-btn {
        display: block;
        width: 100%;
        padding: 12px 16px;
        margin: 4px 0;
        background: transparent;
        border: none;
        color: #94A3B8;
        text-align: left;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }
    .nav-btn.active {
        background: rgba(124, 58, 237, 0.15);
        color: #C084FC;
    }
    
    /* Daily Quote Card (The Neon One) */
    .quote-banner {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(236, 72, 153, 0.15));
        border: 1px solid rgba(236, 72, 153, 0.5);
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.3), inset 0 0 15px rgba(236, 72, 153, 0.2);
        border-radius: 20px;
        padding: 20px 30px;
        text-align: center;
        margin-bottom: 2rem;
        animation: float 4s ease-in-out infinite;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Chat & Workspace Layout */
    .stChatFloatingInputContainer { background: transparent !important; }
    
    .chat-bubble {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        max-width: 85%;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .user-msg {
        background: linear-gradient(135deg, #7C3AED, #4F46E5);
        margin-left: auto;
        border-bottom-right-radius: 4px;
        color: white;
    }
    .assistant-msg {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(124, 58, 237, 0.2);
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    .doc-panel {
        background: #1E293B;
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid rgba(124, 58, 237, 0.1);
        height: 600px;
        overflow-y: auto;
    }
    
    /* Energy Bar */
    .energy-bar-bg {
        background: #334155;
        height: 6px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .energy-bar-fill {
        background: linear-gradient(90deg, #7C3AED, #F472B6);
        height: 100%;
        border-radius: 10px;
        width: 90%; /* Placeholder for 9/10 */
    }
    </style>
""", unsafe_allow_html=True)

# --- State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = "brc"
if "current_panel" not in st.session_state:
    st.session_state.current_panel = "dashboard"
if "energy" not in st.session_state:
    st.session_state.energy = 9
if "mood" not in st.session_state:
    st.session_state.mood = "Pozitif"
if "fired_alarms" not in st.session_state:
    st.session_state.fired_alarms = set()
if "pending_voice_text" not in st.session_state:
    st.session_state.pending_voice_text = None

# --- Sidebar Rebuild ---
with st.sidebar:
    st.markdown('<span class="logo-text">💠 IREM AI</span>', unsafe_allow_html=True)
    
    # Navigation Simulation
    if st.button("🏠 Dashboard", key="btn_dash", use_container_width=True):
        st.session_state.current_panel = "dashboard"
    if st.button("🎯 Hedeflerim", key="btn_goals", use_container_width=True):
        st.session_state.current_panel = "goals"
    if st.button("📅 Takvim", key="btn_cal", use_container_width=True):
        st.session_state.current_panel = "calendar"
    if st.button("📊 Analitik", key="btn_ana", use_container_width=True):
        st.session_state.current_panel = "analytics"
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Energy Widget
    energy_pct = st.session_state.energy * 10
    st.markdown(f"""
        <div style="font-size:0.82rem; color:#94A3B8;">
            <div style="display:flex; justify-content:space-between;">
                <span>Enerji</span>
                <span>{st.session_state.energy}/10</span>
            </div>
            <div class="energy-bar-bg"><div class="energy-bar-fill" style="width:{energy_pct}% !important;"></div></div>
            <p style="color:#10B981; font-size:0.72rem;">✨ Ruh Hali: {st.session_state.mood}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # User Profile
    with st.expander("👤 Profil", expanded=False):
        new_name = st.text_input("Görünen Ad", value=st.session_state.user_name)
        if new_name != st.session_state.user_name:
            st.session_state.user_name = new_name
            st.rerun()

    st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; padding:10px;">
            <div style="width:36px; height:36px; background:#7C3AED; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800;">{st.session_state.user_name[0].upper() if st.session_state.user_name else 'U'}</div>
            <div>
                <p style="font-size:0.88rem; font-weight:600; margin:0;">{st.session_state.user_name}</p>
                <p style="font-size:0.72rem; color:#10B981; margin:0;">● Online</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- ALARM CHECK (Fix 2) ---
def check_alarms():
    conn = sqlite3.connect('goals.db')
    c = conn.cursor()
    today_str = datetime.now().strftime('%Y-%m-%d')
    now_time = datetime.now().strftime('%H:%M')
    
    c.execute("SELECT id, subject, start_time FROM events WHERE date = ?", (today_str,))
    todays_events = c.fetchall()
    conn.close()
    
    for eid, subject, start_time in todays_events:
        alarm_id = f"alarm_{eid}"
        if start_time == now_time and alarm_id not in st.session_state.fired_alarms:
            st.toast(f"⏰ {subject} başlıyor!", icon="🔔")
            st.warning(f"⚠️ **Hatırlatıcı:** {subject} şu an başlıyor ({start_time})")
            st.session_state.fired_alarms.add(alarm_id)

check_alarms()

# --- Main Logic ---
if st.session_state.current_panel == "dashboard":
    # Header
    col_h1, col_h2 = st.columns([1, 1])
    with col_h1:
        st.markdown('<h1 style="margin-bottom:0;">Hoş Geldin! 👋</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94A3B8; margin-top:0;">Koçunla sohbet et, planını oluştur, hedeflerini takip et!</p>', unsafe_allow_html=True)
    
    # Daily Quote Banner
    all_quotes = [q for quotes in QUOTES.values() for q in quotes]
    if "daily_quote" not in st.session_state:
        st.session_state.daily_quote = random.choice(all_quotes)
    
    quote = st.session_state.daily_quote
    st.markdown(f"""
        <div class="quote-banner">
            <div style="font-size:11px; color:#F472B6; font-weight:800; letter-spacing:2px; margin-bottom:10px; text-transform:uppercase;">✨ GÜNÜN SÖZÜ</div>
            <div style="font-size:1.2rem; font-weight:700; color:white; line-height:1.5;">"{quote['text']}"</div>
            <div style="font-size:0.95rem; color:#F472B6; margin-top:10px;">— {quote['author']} {quote['emoji']}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Workspace
    col_chat, col_doc = st.columns([1.1, 1.4], gap="large")
    
    with col_chat:
        # Chat Messages
        chat_sub = st.container(height=500, border=False)
        with chat_sub:
            for m in st.session_state.messages:
                cls = "user-msg" if m["role"] == "user" else "assistant-msg"
                st.markdown(f'<div class="chat-bubble {cls}">{m["content"]}</div>', unsafe_allow_html=True)

        # Chat Input Area (Fix 4 - Voice Input)
        col_mic, col_in = st.columns([1, 10])
        with col_mic:
            audio_bytes = audio_recorder(text="", icon_size="2x", neutral_color="#7C3AED")
        
        if audio_bytes and "last_audio" not in st.session_state or st.session_state.get("last_audio") != audio_bytes:
            st.session_state.last_audio = audio_bytes
            with st.spinner("Ses çözülüyor..."):
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio_bytes)
                try:
                    with open("temp_audio.wav", "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                        st.session_state.pending_voice_text = transcript.text
                except Exception as e:
                    st.error(f"Fısıltı hatası: {e}")

        # Chat Input Area
        prompt = st.chat_input("Zihin koçun burada, hadi konuşalım...")
        if st.session_state.pending_voice_text:
            prompt = st.session_state.pending_voice_text
            st.session_state.pending_voice_text = None

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Agent logic
            config = {"configurable": {"thread_id": "st-session-1"}}
            state_update = {"messages": [HumanMessage(content=prompt)], "user_name": st.session_state.user_name, "language": "tr"}
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_state = loop.run_until_complete(coaching_agent_app.ainvoke(state_update, config))
            
            last_msg = response_state["messages"][-1].content
            active_agent = response_state.get("next_agent", "SYSTEM")
            
            # Update metrics
            st.session_state.energy = response_state.get("energy_score", st.session_state.energy)
            mood_map = {"Positive": "Pozitif", "Neutral": "Nötr", "Tired": "Yorgun", "Anxious": "Endişeli"}
            st.session_state.mood = mood_map.get(response_state.get("sentiment"), st.session_state.mood)
            
            st.session_state.messages.append({"role": "assistant", "content": last_msg, "agent": active_agent})
            st.rerun()

    with col_doc:
        st.markdown('<div class="doc-panel">', unsafe_allow_html=True)
        st.markdown('### Çalışma Alanı')
        
        last_teach = next((m for m in reversed(st.session_state.messages) if m.get("agent") in ["TEACH", "PLAN"]), None)
        if last_teach:
            st.markdown(last_teach["content"])
            # PDF Button
            pdf_data = create_pdf_buffer(last_teach["content"], f"IREM AI - {last_teach['agent']}")
            st.download_button("📄 PDF AI", data=pdf_data, file_name="irem_not.pdf", mime="application/pdf")
        else:
            st.markdown("*Ders notları ve testler burada görünecek.*")
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_panel == "goals":
    st.markdown("## 🎯 Hedeflerim")
    st.markdown("Başarmak istediğin hedefleri ekle ve takip et.")
    
    # Local DB Helpers
    def add_goal_local(title, desc, start, end):
        conn = sqlite3.connect('goals.db')
        c = conn.cursor()
        c.execute("INSERT INTO goals (title, description, start_date, end_date, status) VALUES (?, ?, ?, ?, ?)",
                  (title, desc, start, end, 'Devam Ediyor'))
        conn.commit()
        conn.close()

    def get_goals_local():
        conn = sqlite3.connect('goals.db')
        c = conn.cursor()
        c.execute("SELECT * FROM goals ORDER BY created_at DESC")
        rows = c.fetchall()
        conn.close()
        return rows

    def mark_done_local(gid):
        conn = sqlite3.connect('goals.db')
        c = conn.cursor()
        c.execute("UPDATE goals SET status = 'Tamamlandı' WHERE id = ?", (gid,))
        conn.commit()
        conn.close()

    def delete_goal_local(gid):
        conn = sqlite3.connect('goals.db')
        c = conn.cursor()
        c.execute("DELETE FROM goals WHERE id = ?", (gid,))
        conn.commit()
        conn.close()

    # Add Goal Form (Fix 1)
    with st.expander("➕ Yeni Hedef Ekle", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Hedef Başlığı", placeholder="Ne başarmak istiyorsun?")
            start_date = st.date_input("Başlangıç Tarihi", key="g_start")
        with col2:
            subject = st.selectbox("Ders / Alan", ["Matematik", "Fen", "Türkçe", "İngilizce", "Diğer"])
            end_date = st.date_input("Bitiş Tarihi", key="g_end")
        
        if st.button("Kaydet", use_container_width=True):
            if title:
                add_goal_local(title, subject, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                st.success("Hedef eklendi!")
                st.rerun()
            else:
                st.error("Lütfen bir başlık girin.")

    # List Goals
    goals = get_goals_local()
    if not goals:
        st.info("Henüz hedef eklemedin 🎯")
    else:
        cols = st.columns(3)
        for i, goal in enumerate(goals):
            gid, gtitle, gdesc, gstart, gend, gstatus, gcreated = goal
            with cols[i % 3]:
                status_color = "#10B981" if gstatus == "Tamamlandı" else "#F59E0B"
                st.markdown(f"""
                <div style="background:#1E293B; padding:1.2rem; border-radius:18px; border:1px solid rgba(124, 58, 237, 0.3); margin-bottom:1rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span style="background:rgba(124,58,237,0.2); color:#A78BFA; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:700;">{gdesc}</span>
                        <span style="background:{status_color}22; color:{status_color}; font-size:0.65rem; padding:2px 8px; border-radius:10px; font-weight:800;">{gstatus}</span>
                    </div>
                    <h4 style="margin:10px 0;">{gtitle}</h4>
                    <p style="font-size:0.78rem; color:#94A3B8;">📅 {gstart} ➔ {gend}</p>
                </div>
                """, unsafe_allow_html=True)
                
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if gstatus != "Tamamlandı":
                        if st.button(f"Tamamla", key=f"done_{gid}"):
                            mark_done_local(gid)
                            st.balloons()
                            st.rerun()
                with b_col2:
                    if st.button(f"Sil", key=f"del_{gid}"):
                        delete_goal_local(gid)
                        st.rerun()

elif st.session_state.current_panel == "calendar":
    st.markdown("## 📅 Takvim")
    st.markdown("Çalışma seanslarını kaydet ve günlük ilerlemeyi gör.")
    
    # Local DB Helpers
    def add_event_local(subject, date, start, end, duration):
        conn = sqlite3.connect('goals.db')
        c = conn.cursor()
        c.execute("INSERT INTO events (subject, date, start_time, end_time, duration) VALUES (?, ?, ?, ?, ?)",
                  (subject, date, start, end, duration))
        conn.commit()
        conn.close()

    def get_events_local():
        conn = sqlite3.connect('goals.db')
        c = conn.cursor()
        c.execute("SELECT * FROM events")
        rows = c.fetchall()
        conn.close()
        return rows

    # Add Session Form (Fix 2)
    with st.expander("📝 Çalışma Kaydı Ekle", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            subject = st.selectbox("Ders", ["Matematik", "Fen", "Türkçe", "İngilizce", "Diğer"])
            date = st.date_input("Tarih")
        with col2:
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                start_t = st.time_input("Başlangıç", datetime.now().time())
            with t_col2:
                end_t = st.time_input("Bitiş", (datetime.now() + timedelta(minutes=45)).time())
            duration = st.number_input("Süre (dk)", min_value=1, max_value=480, value=45)
        
        if st.button("Ekle", use_container_width=True):
            add_event_local(subject, date.strftime('%Y-%m-%d'), start_t.strftime('%H:%M'), end_t.strftime('%H:%M'), duration)
            st.success("Kayıt eklendi!")
            st.rerun()

    # Weekly Grid
    st.markdown("### Bu Hafta")
    sessions = get_events_local()
    
    # Calculate week dates
    today = datetime.now()
    start_of_week = today - timedelta(days=today.weekday())
    week_days = [start_of_week + timedelta(days=i) for i in range(7)]
    day_names = ["Pzt", "Sal", "Çrş", "Prş", "Cum", "Cmt", "Paz"]
    
    cols = st.columns(7)
    for i, day in enumerate(week_days):
        with cols[i]:
            is_today = day.date() == today.date()
            color = "#7C3AED" if is_today else "#94A3B8"
            st.markdown(f"""
            <div style="text-align:center; margin-bottom:10px;">
                <div style="font-size:0.8rem; color:#94A3B8;">{day_names[i]}</div>
                <div style="font-size:1.2rem; font-weight:700; color:{color};">{day.day}</div>
            </div>
            """, unsafe_allow_html=True)
            
            day_sessions = [s for s in sessions if s[2] == day.strftime('%Y-%m-%d')]
            for s in day_sessions:
                st.markdown(f"""
                <div style="background:rgba(124,58,237,0.1); border:1px solid rgba(124,58,237,0.2); border-radius:6px; padding:4px; font-size:0.65rem; margin-bottom:4px; text-align:center; color:#C4B5FD;">
                    {s[1]}<br>{s[3]} - {s[4]}<br>{s[5]}dk
                </div>
                """, unsafe_allow_html=True)

elif st.session_state.current_panel == "analytics":
    st.markdown("## 📊 Analitik")
    st.markdown("Performansını grafiklerle incele.")
    
    db = get_db()
    user_id = 1
    sessions = crud.get_study_sessions(db, user_id)
    
    # Fetch local goals (Fix 3)
    conn = sqlite3.connect('goals.db')
    local_goals = pd.read_sql_query("SELECT * FROM goals", conn)
    conn.close()
    
    # Goal Metrics
    total_goals = len(local_goals)
    completed_goals = len(local_goals[local_goals['status'] == 'Tamamlandı'])
    completion_rate = (completed_goals / total_goals * 100) if total_goals > 0 else 0
    
    # Study Metrics
    today = datetime.now().date()
    week_start = today - timedelta(days=today.weekday())
    week_mins = sum(s.duration_minutes for s in sessions if s.date.date() >= week_start)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Bu Hafta (Saat)", f"{week_mins/60:.1f}")
    with col2:
        st.metric("Toplam Hedef", total_goals)
    with col3:
        st.metric("Tamamlanan", completed_goals)
    with col4:
        st.metric("Tamamlama Oranı", f"%{completion_rate:.0f}")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### Son 7 Gün Aktivite")
        last_7_days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
        chart_data = []
        for d in last_7_days:
            mins = sum(s.duration_minutes for s in sessions if s.date.date() == d)
            chart_data.append({"Gün": d.strftime('%d %b'), "Dakika": mins})
        
        df_activity = pd.DataFrame(chart_data)
        st.bar_chart(df_activity.set_index("Gün"))

    with col_chart2:
        st.markdown("### Hedef Durumu")
        if total_goals > 0:
            status_df = local_goals['status'].value_counts().reset_index()
            status_df.columns = ['Durum', 'Sayı']
            fig = px.pie(status_df, values='Sayı', names='Durum', 
                         color_discrete_map={'Tamamlandı':'#10B981', 'Devam Ediyor':'#F59E0B'},
                         hole=.4)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                              font_color='white', margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Grafik için henüz hedef verisi yok.")
